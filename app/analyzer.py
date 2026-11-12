"""Delta Lake 版本可恢复性分析核心逻辑。

值班员上传 ``_delta_log`` 中留存的 JSON 提交（``NNN.json``）、经典单文件
Parquet 检查点（``NNN.checkpoint.parquet``）以及现存数据文件的表内相对路径
清单。本模块逐版重放提交 / 检查点，重建每个版本快照仍引用的数据文件，
再与现存清单核对，判断该版本能否完整重建。

仅支持**不包含删除向量（Deletion Vector）**的表；一旦在协议特性或任意
add 动作中发现删除向量，立即拒绝分析并提示文件来源。
"""

from __future__ import annotations

import io
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

# Delta 日志文件名均为 20 位补零版本号
_COMMIT_RE = re.compile(r"^(\d{20})\.json$")
# 经典单文件检查点：00000000000000000010.checkpoint.parquet
_CHECKPOINT_RE = re.compile(r"^(\d{20})\.checkpoint\.parquet$")
# 多文件（分片）检查点，本工具不支持，仅登记并忽略
_MULTI_CHECKPOINT_RE = re.compile(r"^\d{20}\.checkpoint\.\d+\.\d+\.parquet$")

DV_FEATURE_NAME = "deletionVectors"

STATUS_RECOVERABLE = "recoverable"
STATUS_MISSING_COMMIT = "missing_commit"
STATUS_MISSING_FILES = "missing_files"
# 重放完整，但清单格式未注明，同一行既可解释为原样路径、也可解释为
# URI 解码后的路径；这些引用文件既不能算现存，也不能直接判为缺失。
STATUS_PENDING_CONFIRMATION = "pending_confirmation"

# 清单（manifest）路径格式：
#   raw  —— 对象存储导出的原样文件名，不做任何 percent-decode
#   uri  —— 路径经过 URI 编码，恰好解码一次
#   auto —— 旧清单未注明格式：原样与解码结果相同时才可确认；不同则待确认
PATH_FORMAT_RAW = "raw"
PATH_FORMAT_URI = "uri"
PATH_FORMAT_AUTO = "auto"

_FORMAT_ALIASES = {
    "raw": PATH_FORMAT_RAW,
    "literal": PATH_FORMAT_RAW,
    "plain": PATH_FORMAT_RAW,
    "as-is": PATH_FORMAT_RAW,
    "asis": PATH_FORMAT_RAW,
    "unencoded": PATH_FORMAT_RAW,
    "原样": PATH_FORMAT_RAW,
    "原样路径": PATH_FORMAT_RAW,
    "原始": PATH_FORMAT_RAW,
    "原始路径": PATH_FORMAT_RAW,
    "uri": PATH_FORMAT_URI,
    "url": PATH_FORMAT_URI,
    "uri编码": PATH_FORMAT_URI,
    "uri编码路径": PATH_FORMAT_URI,
    "url编码": PATH_FORMAT_URI,
    "percent": PATH_FORMAT_URI,
    "percentencoded": PATH_FORMAT_URI,
    "percent-encoding": PATH_FORMAT_URI,
    "percent编码": PATH_FORMAT_URI,
    "%编码": PATH_FORMAT_URI,
    "百分号编码": PATH_FORMAT_URI,
    "encoded": PATH_FORMAT_URI,
    "urlencoded": PATH_FORMAT_URI,
    "auto": PATH_FORMAT_AUTO,
    "automatic": PATH_FORMAT_AUTO,
    "unknown": PATH_FORMAT_AUTO,
    "unspecified": PATH_FORMAT_AUTO,
    "legacy": PATH_FORMAT_AUTO,
    "自动": PATH_FORMAT_AUTO,
    "未注明": PATH_FORMAT_AUTO,
    "未注明格式": PATH_FORMAT_AUTO,
    "未标注": PATH_FORMAT_AUTO,
    "未指定": PATH_FORMAT_AUTO,
    "未指定格式": PATH_FORMAT_AUTO,
    "不确定": PATH_FORMAT_AUTO,
    "旧清单": PATH_FORMAT_AUTO,
}

# 清单首部注释里的格式声明，例如：
#   # format: raw
#   # path-format: uri
#   # 格式：原样
_FORMAT_DIRECTIVE_RE = re.compile(
    r"^#\s*(format|path[-_ ]?format|manifest[-_ ]?format|encoding"
    r"|格式|路径格式|清单格式|编码)\s*[:：=]\s*(.*?)\s*$",
    re.IGNORECASE,
)

# 同一字段既可能声明格式（format / 格式），也可能是“编码：UTF-8”之类的
# 注释；只有取值能识别成路径格式时才当作格式声明。
_FORMAT_KEY_FEATURE = re.compile(r"format|格式", re.IGNORECASE)

# 文件名后缀也可以声明格式，例如 object_listing.raw.txt / backup.uri.txt；
# 旧清单（existing_files.txt）不命中，保持 auto。要求格式词前后都有分隔符，
# 避免把 raw_data.txt 这类普通名字误当成格式声明。
_RAW_FILENAME_RE = re.compile(
    r"[._-](raw|literal|plain|as[-_]is)(?:[._-]|$)", re.IGNORECASE)
_URI_FILENAME_RE = re.compile(
    r"[._-](uri|url|encoded|percent[-_]encoded)(?:[._-]|$)", re.IGNORECASE)


class AnalyzerError(Exception):
    """面向用户的输入 / 能力限制错误。"""


@dataclass(init=False)
class InputFile:
    """一次上传的文件（文件名 + 已读入内存的字节）。

    ``path_format`` 仅对路径清单有意义，可选 ``raw`` / ``uri`` / ``auto``
    （或 ``None`` 表示未注明）。为便于调用方沿用旧代码，也接受关键字
    别名 ``format``。
    """

    filename: str
    data: bytes
    path_format: str | None = None

    def __init__(self, filename: str, data: bytes,
                 path_format: str | None = None,
                 format: str | None = None) -> None:
        self.filename = filename
        self.data = data
        self.path_format = path_format if path_format is not None else format


@dataclass
class ParsedCommit:
    version: int
    source: str
    adds: list[tuple[str, dict]] = field(default_factory=list)
    removes: list[str] = field(default_factory=list)
    dv_detected: bool = False
    protocol_features: list[str] = field(default_factory=list)


@dataclass
class ParsedCheckpoint:
    version: int
    source: str
    adds: list[str] = field(default_factory=list)
    dv_detected: bool = False
    protocol_features: list[str] = field(default_factory=list)
    tombstones: list[str] = field(default_factory=list)


def _strip_table_prefix(path: str) -> str:
    """去掉表内相对路径常见的前导 ``./`` 与 ``/``。"""
    while path.startswith("./"):
        path = path[2:]
    while path.startswith("/"):
        path = path[1:]
    return path


def decode_uri_path_once(raw: Any, source_description: str = "路径") -> str:
    """URI 编码路径只解码一次（``%2520`` → ``%20``，而不是空格）。

    解码后仍会去掉前导 ``./`` / ``/``；不会对结果再次解码，避免把文件名
    中字面的 ``%20`` 误当成空格。
    """
    if not isinstance(raw, str):
        raise AnalyzerError(f"{source_description}不是字符串：{raw!r}")
    return _strip_table_prefix(urllib.parse.unquote(raw.strip()))


def normalize_raw_path(raw: Any, source_description: str = "路径") -> str:
    """原样路径：不做任何 percent-decode，仅去掉前导 ``./`` / ``/``。"""
    if not isinstance(raw, str):
        raise AnalyzerError(f"{source_description}不是字符串：{raw!r}")
    return _strip_table_prefix(raw.strip())


def normalize_data_path(raw: Any) -> str:
    """把 Delta 日志 / 检查点中的路径归一化为表内相对路径。

    JSON 提交和经典单文件 Parquet 检查点中的路径都是 URI 编码的，统一
    恰好解码一次（``%2520`` 只会变成 ``%20``）。文件名中字面的 ``%20``
    若要还原为空格，由上传方在清单里显式声明 URI 编码 —— 日志侧固定按
    规范解码一次，不会循环解码。
    """
    return decode_uri_path_once(raw, "数据文件路径")


def _decode_utf8(data: bytes, source: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalyzerError(f"{source} 不是合法的 UTF-8 文本：{exc}") from exc


def _iter_json_objects(text: str, source: str):
    """Delta 提交是 ndjson（每行一个动作对象），同时兼容整体 JSON 数组。"""
    stripped = text.strip()
    if not stripped:
        return
    # 先完整解析 ndjson；整行恰好是 JSON 数组 / 对象，或任意行失败则整体回退，
    # 避免把单行 JSON 数组误当成 ndjson、或吐出半截动作
    line_objs = []
    ndjson_ok = True
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            ndjson_ok = False
            break
        if not isinstance(obj, dict):
            ndjson_ok = False
            break
        line_objs.append(obj)
    if ndjson_ok and line_objs:
        yield from line_objs
        return
    try:
        whole = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"{source} 既不是 ndjson 也不是合法 JSON：{exc}") from exc
    if isinstance(whole, list):
        yield from whole
    else:
        yield whole


def _extract_protocol_features(protocol: dict) -> list[str]:
    features: list[str] = []
    for key in ("readerFeatures", "writerFeatures", "features"):
        value = protocol.get(key)
        if isinstance(value, list):
            features.extend(str(v) for v in value)
    return features


def _add_has_dv(add_action: dict) -> bool:
    dv = add_action.get("deletionVector")
    return dv is not None


def parse_commit(version: int, data: bytes, source: str) -> ParsedCommit:
    """解析一个 JSON 提交文件。"""
    parsed = ParsedCommit(version=version, source=source)
    text = _decode_utf8(data, source)
    for obj in _iter_json_objects(text, source):
        if not isinstance(obj, dict):
            continue
        if "add" in obj and isinstance(obj["add"], dict):
            add = obj["add"]
            path = normalize_data_path(add.get("path"))
            parsed.adds.append((path, add))
            if _add_has_dv(add):
                parsed.dv_detected = True
        if "remove" in obj and isinstance(obj["remove"], dict):
            remove = obj["remove"]
            parsed.removes.append(normalize_data_path(remove.get("path")))
            if remove.get("deletionVector") is not None:
                parsed.dv_detected = True
        if "protocol" in obj and isinstance(obj["protocol"], dict):
            parsed.protocol_features.extend(
                _extract_protocol_features(obj["protocol"])
            )
    return parsed


def parse_checkpoint(version: int, data: bytes, source: str) -> ParsedCheckpoint:
    """用 PyArrow 解析经典单文件 Parquet 检查点。"""
    parsed = ParsedCheckpoint(version=version, source=source)
    try:
        table = pq.read_table(io.BytesIO(data))
    except Exception as exc:  # parquet 损坏 / 非 parquet
        raise AnalyzerError(f"{source} 无法作为 Parquet 检查点读取：{exc}") from exc

    column_names = set(table.column_names)
    if "add" not in column_names:
        raise AnalyzerError(f"{source} 中缺少 add 列，不是合法的 Delta 检查点")

    # Table.column 返回 ChunkedArray，需 combine_chunks 后才能按 struct 取子字段
    add_col = table.column("add").combine_chunks()
    add_type = add_col.type
    n = table.num_rows
    # 以列类型而非逐行值判断是否带 DV：全 null 行也能正确识别
    add_has_dv_column = pa.types.is_struct(add_type) and \
        (add_type.get_field_index("deletionVector") != -1)
    if add_has_dv_column:
        dv_values = add_col.field("deletionVector").to_pylist()
    else:
        dv_values = [None] * n
    paths = add_col.field("path").to_pylist()
    for path, dv in zip(paths, dv_values):
        if path is None:
            continue
        parsed.adds.append(normalize_data_path(path))
        if dv is not None:
            parsed.dv_detected = True

    # 检查点里的 remove 是墓碑（tombstone），不属于当前快照，
    # 仅记录用于展示，重放基线时不据此删除文件。
    if "remove" in column_names:
        remove_col = table.column("remove").combine_chunks()
        for path in remove_col.field("path").to_pylist():
            if path is not None:
                parsed.tombstones.append(normalize_data_path(path))
        if pa.types.is_struct(remove_col.type) and \
                remove_col.type.get_field_index("deletionVector") != -1:
            if any(v is not None for v in
                   remove_col.field("deletionVector").to_pylist()):
                parsed.dv_detected = True

    if "protocol" in column_names:
        proto_col = table.column("protocol").combine_chunks()
        for field_name in ("readerFeatures", "writerFeatures", "features"):
            if pa.types.is_struct(proto_col.type) and \
                    proto_col.type.get_field_index(field_name) != -1:
                for value in proto_col.field(field_name).to_pylist():
                    if isinstance(value, list):
                        parsed.protocol_features.extend(str(v) for v in value)
    return parsed


@dataclass
class ManifestEntry:
    """清单中的一行路径及其解释方式。"""

    raw: str
    decoded: str
    path_format: str
    source: str
    line_no: int

    @property
    def ambiguous(self) -> bool:
        return self.raw != self.decoded


def _canonical_format(value: Any, source: str) -> str | None:
    """把声明值归一为 raw / uri / auto；无法识别时返回 None。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnalyzerError(f"{source}：路径格式必须是字符串，收到 {value!r}")
    # 声明行后面的说明文字（如 “raw —— 原样路径，不解码”）不属于格式值
    token = re.split(r"[\s，,；;（()）—]+", value.strip(), maxsplit=1)[0]
    key = token.strip().lower().replace(" ", "").replace("_", "-")
    canonical = _FORMAT_ALIASES.get(key)
    if canonical is None:
        # 再试一次去掉连字符的写法（percent-encoded / percentencoded）
        canonical = _FORMAT_ALIASES.get(key.replace("-", ""))
    return canonical


def _require_format(value: Any, source: str) -> str:
    canonical = _canonical_format(value, source)
    if canonical is None:
        raise AnalyzerError(
            f"{source}：无法识别的路径格式 {value!r}，"
            f"应为 raw（原样路径）、uri（URI 编码）或 auto（未注明/自动判断）"
        )
    return canonical


def _format_from_filename(filename: str) -> str:
    """从文件名推断清单格式；无法判断时按旧清单的 auto 处理。"""
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if _RAW_FILENAME_RE.search(stem):
        return PATH_FORMAT_RAW
    if _URI_FILENAME_RE.search(stem):
        return PATH_FORMAT_URI
    return PATH_FORMAT_AUTO


def _format_directive(line: str) -> tuple[str | None, str | None]:
    """解析清单首部注释里的格式声明，返回 (key, value)。"""
    m = _FORMAT_DIRECTIVE_RE.match(line)
    if not m:
        return None, None
    return m.group(1).strip().lower().replace(" ", ""), m.group(2)


def _resolve_manifest_format(data: bytes, filename: str,
                             declared: str | None,
                             text: str) -> str:
    """确定一份清单的路径格式。

    优先级：上传参数/表单显式声明 > 清单首部 ``# format:`` 指令 >
    文件名后缀提示 > auto（旧清单未注明格式）。
    """
    if declared is not None and str(declared).strip() != "":
        return _require_format(declared, f"清单 {filename}")
    for line in text.splitlines():
        key, value = _format_directive(line)
        if key is None:
            continue
        if not value:
            if _FORMAT_ALIASES.get(key) is not None:
                return _FORMAT_ALIASES[key]
            continue
        if not _FORMAT_KEY_FEATURE.search(key):
            # 形如 “# 编码：UTF-8”，不是路径格式声明
            continue
        canonical = _canonical_format(value, f"清单 {filename} 的格式声明")
        if canonical is not None:
            return canonical
    return _format_from_filename(filename)


def parse_manifest_entries(data: bytes, source: str,
                           path_format: str | None = None,
                           format: str | None = None
                           ) -> tuple[list[ManifestEntry], str]:
    """解析路径清单，保留下每行的原样与解码结果。

    返回 ``(entries, resolved_format)``。清单格式为 raw 时逐行原样保留；
    为 uri 时每行只解码一次；为 auto（旧清单未注明）时，原样与解码结果
    相同的行无歧义，不同的行保留两种候选，由调用方标为待确认。
    ``format`` 是 ``path_format`` 的别名，方便调用方按 JSON 字段名传参。
    """
    if path_format is None and format is not None:
        path_format = format
    text = _decode_utf8(data, source)
    resolved = _resolve_manifest_format(data, source, path_format, text)
    entries: list[ManifestEntry] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw = normalize_raw_path(stripped, f"{source} 第 {line_no} 行")
        decoded = decode_uri_path_once(stripped, f"{source} 第 {line_no} 行")
        entries.append(ManifestEntry(
            raw=raw,
            decoded=decoded,
            path_format=resolved,
            source=source,
            line_no=line_no,
        ))
    return entries, resolved


def parse_manifest(data: bytes, source: str,
                   path_format: str | None = None,
                   format: str | None = None) -> set[str]:
    """解析路径清单并返回所有可能指向的文件路径（向后兼容的集合接口）。

    需要区分“确认现存 / 待确认”时请使用 :func:`parse_manifest_entries`。
    """
    entries, _ = parse_manifest_entries(data, source, path_format, format)
    paths: set[str] = set()
    for entry in entries:
        paths.add(entry.raw)
        paths.add(entry.decoded)
    return paths


def _classify_manifest_entry(entry: ManifestEntry,
                             confirmed: set[str],
                             pending: dict[str, list[dict[str, Any]]]) -> None:
    """把一条清单记录归入“确认现存”或“待确认”。"""
    evidence = {"source": entry.source, "line": entry.line_no,
                "raw": entry.raw, "decoded": entry.decoded,
                "format": entry.path_format}
    if entry.path_format == PATH_FORMAT_RAW:
        confirmed.add(entry.raw)
    elif entry.path_format == PATH_FORMAT_URI:
        confirmed.add(entry.decoded)
    elif not entry.ambiguous:
        confirmed.add(entry.raw)
    else:
        pending.setdefault(entry.raw, []).append(evidence)
        pending.setdefault(entry.decoded, []).append(evidence)


def _classify_logs(log_files: list[InputFile]):
    commits: dict[int, ParsedCommit] = {}
    checkpoints: dict[int, ParsedCheckpoint] = {}
    ignored: list[dict[str, str]] = []
    duplicate: list[str] = []

    for item in log_files:
        name = item.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        m_commit = _COMMIT_RE.match(name)
        m_ckpt = _CHECKPOINT_RE.match(name)
        if m_commit:
            version = int(m_commit.group(1))
            if version in commits:
                duplicate.append(name)
                continue
            commits[version] = parse_commit(version, item.data, name)
        elif m_ckpt:
            version = int(m_ckpt.group(1))
            if version in checkpoints:
                duplicate.append(name)
                continue
            checkpoints[version] = parse_checkpoint(version, item.data, name)
        elif _MULTI_CHECKPOINT_RE.match(name):
            ignored.append({"name": name,
                            "reason": "分片（多文件）检查点，本工具仅支持经典单文件检查点"})
        elif name == "_last_checkpoint":
            ignored.append({"name": name, "reason": "检查点指针文件，不参与重放"})
        else:
            ignored.append({"name": name, "reason": "无法识别的 _delta_log 文件名"})
    return commits, checkpoints, ignored, duplicate


def _guard_deletion_vectors(commits: dict[int, ParsedCommit],
                            checkpoints: dict[int, ParsedCheckpoint]) -> None:
    for version, parsed in sorted(commits.items()):
        if DV_FEATURE_NAME in parsed.protocol_features:
            raise AnalyzerError(
                f"表启用了删除向量：提交 {parsed.source} 的 protocol 特性包含 "
                f"{DV_FEATURE_NAME}。本工具只处理无删除向量的表。"
            )
        if parsed.dv_detected:
            raise AnalyzerError(
                f"表包含删除向量：提交 {parsed.source} 的 add/remove 动作带有 "
                f"deletionVector 字段。本工具只处理无删除向量的表。"
            )
    for version, parsed in sorted(checkpoints.items()):
        if DV_FEATURE_NAME in parsed.protocol_features:
            raise AnalyzerError(
                f"表启用了删除向量：检查点 {parsed.source} 的 protocol 特性包含 "
                f"{DV_FEATURE_NAME}。本工具只处理无删除向量的表。"
            )
        if parsed.dv_detected:
            raise AnalyzerError(
                f"表包含删除向量：检查点 {parsed.source} 的 add 动作带有非空 "
                f"deletionVector 结构。本工具只处理无删除向量的表。"
            )


def _choose_base(target: int,
                 commits: dict[int, ParsedCommit],
                 checkpoints: dict[int, ParsedCheckpoint]):
    """选择重放起点。

    优先使用不晚于目标版本的最新检查点（重放距离最短）；
    没有可用检查点时，从版本 0 的提交出发。
    返回 (base_kind, base_version) 或 (None, None)。
    """
    eligible = [v for v in checkpoints if v <= target]
    if eligible:
        base_version = max(eligible)
        return "checkpoint", base_version
    if 0 in commits:
        return "commit", 0
    return None, None


def _replay(target: int,
            commits: dict[int, ParsedCommit],
            checkpoints: dict[int, ParsedCheckpoint]) -> dict[str, Any]:
    """对单个目标版本执行一次完整重放并核对快照引用文件。"""
    base_kind, base_version = _choose_base(target, commits, checkpoints)
    missing_commits: list[int] = []
    active: dict[str, dict] = {}

    if base_kind is None:
        # 连起点都没有：版本 0 提交与任何检查点均缺失
        missing_commits = list(range(0, target + 1))
        return {
            "base": None,
            "missing_commits": missing_commits,
            "active": active,
            "replay_complete": False,
            "note": "缺少重放起点：版本 0 的提交与不晚于该版本的检查点均未上传",
        }

    if base_kind == "checkpoint":
        ckpt = checkpoints[base_version]
        for path in ckpt.adds:
            active[path] = {"from": f"checkpoint-{base_version}"}
        required_commits = range(base_version + 1, target + 1)
        base_label = f"checkpoint-{base_version}"
    else:
        seed = commits[0]
        for path, meta in seed.adds:
            active[path] = {"from": "commit-0"}
        required_commits = range(1, target + 1)
        base_label = "commit-0"

    for version in required_commits:
        commit = commits.get(version)
        if commit is None:
            missing_commits.append(version)
            continue
        # 同一提交内按文件中的动作顺序应用；remove 先于 add 生效
        for path in commit.removes:
            active.pop(path, None)
        for path, meta in commit.adds:
            active[path] = {"from": f"commit-{version}"}

    return {
        "base": base_label,
        "missing_commits": missing_commits,
        "active": active,
        "replay_complete": not missing_commits,
        "note": None,
    }


def _manifest_format_override(
        index: int,
        formats: Any,
        global_format: str | None) -> str | None:
    """取出第 index 份清单的格式声明。"""
    if formats is None:
        return global_format
    if isinstance(formats, dict):
        # 允许按文件名或上传顺序下标指定
        for key in (index, str(index)):
            if key in formats:
                return formats[key]
    elif isinstance(formats, (list, tuple)):
        if index < len(formats) and formats[index] is not None:
            return formats[index]
    else:
        raise AnalyzerError("manifest_formats 必须是列表或字典")
    return global_format


def _read_manifest_sources(manifest_files: Iterable[InputFile],
                           formats: Any = None,
                           global_format: str | None = None
                           ) -> tuple[list[tuple[str, bytes, str | None]],
                                      list[str]]:
    """把上传清单统一成 (文件名, 字节, 路径格式) 三元组。"""
    read: list[tuple[str, bytes, str | None]] = []
    names: list[str] = []
    for index, item in enumerate(manifest_files):
        if not isinstance(item, InputFile):
            raise AnalyzerError(f"无法识别的清单条目：{item!r}")
        name = item.filename
        data = item.data
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, (bytes, bytearray)):
            raise AnalyzerError(f"清单 {name} 的内容必须是字符串或字节")
        fmt = item.path_format
        if fmt is None:
            fmt = _manifest_format_override(index, formats, global_format)
        base_name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        if isinstance(fmt, str) and not fmt.strip():
            fmt = None
        names.append(base_name)
        read.append((base_name, bytes(data),
                     None if fmt is None else str(fmt)))
    return read, names


def analyze(log_files: list[InputFile],
            manifest_files: list[InputFile],
            from_version: int,
            to_version: int,
            manifest_formats: Any = None,
            manifest_format: str | None = None) -> dict[str, Any]:
    """分析入口：逐版重放并核对现存数据文件。

    清单格式按以下优先级确定：
    1. ``manifest_formats``（按上传顺序的列表 / 字典）或全局
       ``manifest_format``；
    2. :class:`InputFile` 的 ``path_format``；
    3. 清单首部 ``# format: raw|uri`` 指令；
    4. 文件名提示（如 ``export.raw.txt`` / ``backup.uri.txt``）；
    5. 以上都没有时按旧清单的 auto：原样与解码结果不同则两条都待确认。
    """
    if not isinstance(from_version, int) or not isinstance(to_version, int):
        raise AnalyzerError("版本号必须是整数")
    if from_version < 0 or to_version < 0:
        raise AnalyzerError("版本号不能为负数")
    if from_version > to_version:
        raise AnalyzerError("版本区间起点不能大于终点")
    if not log_files:
        raise AnalyzerError("请至少上传一个 _delta_log 中的 JSON 提交或 Parquet 检查点")

    commits, checkpoints, ignored, duplicate = _classify_logs(log_files)
    if not commits and not checkpoints:
        raise AnalyzerError(
            "上传文件中没有可识别的 Delta 日志（应为 20 位版本号命名的 "
            ".json 提交或 .checkpoint.parquet 检查点）"
        )

    _guard_deletion_vectors(commits, checkpoints)

    manifest_sources, manifest_names = _read_manifest_sources(
        manifest_files or [], manifest_formats, manifest_format)
    confirmed: set[str] = set()
    pending: dict[str, list[dict[str, Any]]] = {}
    manifest_info: list[dict[str, Any]] = []
    for name, data, fmt in manifest_sources:
        entries, resolved = parse_manifest_entries(data, name, fmt)
        for entry in entries:
            _classify_manifest_entry(entry, confirmed, pending)
        manifest_info.append({
            "name": name,
            "path_format": resolved,
            "entry_count": len(entries),
            "ambiguous_count": sum(1 for e in entries if e.ambiguous),
        })

    # 有明确条目证实存在的文件仍算现存（覆盖待确认条目可能造成的误伤）
    for path in confirmed:
        pending.pop(path, None)
    ambiguous_paths = set(pending)

    version_reports = []
    summary = {STATUS_RECOVERABLE: 0,
               STATUS_MISSING_COMMIT: 0,
               STATUS_MISSING_FILES: 0,
               STATUS_PENDING_CONFIRMATION: 0}

    for target in range(from_version, to_version + 1):
        result = _replay(target, commits, checkpoints)
        active: dict[str, dict] = result["active"]
        referenced = sorted(active.keys())
        missing_files = sorted(p for p in referenced
                               if p not in confirmed and p not in ambiguous_paths)
        pending_files = sorted(p for p in referenced
                               if p not in confirmed and p in ambiguous_paths)

        if result["missing_commits"]:
            status = STATUS_MISSING_COMMIT
        elif missing_files:
            # 已经有文件被明确证实不存在，该版必然不可回退
            status = STATUS_MISSING_FILES
        elif pending_files:
            # 既不能证明现存，也不能判为缺失，先要求人工确认
            status = STATUS_PENDING_CONFIRMATION
        else:
            status = STATUS_RECOVERABLE
        summary[status] += 1

        file_details = []
        for path in referenced:
            if path in confirmed:
                state = "confirmed"
            elif path in ambiguous_paths:
                state = "pending"
            else:
                state = "missing"
            detail = {"path": path, "state": state}
            if state == "pending":
                detail["evidence"] = pending[path]
            file_details.append(detail)

        version_reports.append({
            "version": target,
            "status": status,
            "replay_complete": result["replay_complete"],
            "base": result["base"],
            "referenced_files": referenced,
            "referenced_file_count": len(referenced),
            "missing_files": missing_files,
            "pending_files": pending_files,
            "file_details": file_details,
            "missing_commits": result["missing_commits"],
            "note": result["note"],
        })

    return {
        "supported": True,
        "range": {"from": from_version, "to": to_version},
        "available_commits": sorted(commits.keys()),
        "available_checkpoints": sorted(checkpoints.keys()),
        "ignored_files": ignored,
        "duplicate_files": sorted(duplicate),
        "manifests": manifest_names,
        "manifest_info": manifest_info,
        "existing_file_count": len(confirmed),
        "pending_file_count": len(ambiguous_paths),
        "summary": summary,
        "versions": version_reports,
    }
