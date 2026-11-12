"""analyzer 核心逻辑测试。

运行（仓库根目录）：
    /tmp/dvenv/bin/python -I -m pytest tests -q
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.analyzer import (
    AnalyzerError,
    InputFile,
    STATUS_MISSING_COMMIT,
    STATUS_MISSING_FILES,
    STATUS_PENDING_CONFIRMATION,
    STATUS_RECOVERABLE,
    analyze,
    normalize_data_path,
    parse_checkpoint,
    parse_commit,
)


# ---------- 构造工具 ----------

ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("size", pa.int64()),
    ("modificationTime", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
ADD_DV_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionVector", pa.struct([
        ("storageType", pa.string()),
        ("pathOrInlineDv", pa.string()),
    ])),
])


def commit_file(version: int, actions: list[dict]) -> InputFile:
    payload = "\n".join(json.dumps(a) for a in actions)
    return InputFile(filename=f"{version:020d}.json",
                     data=payload.encode("utf-8"))


def add(path: str, dv: dict | None = None) -> dict:
    action = {"add": {"path": path, "size": 1, "dataChange": True}}
    if dv is not None:
        action["add"]["deletionVector"] = dv
    return action


def remove(path: str, dv: dict | None = None) -> dict:
    action = {"remove": {"path": path, "dataChange": True}}
    if dv is not None:
        action["remove"]["deletionVector"] = dv
    return action


def checkpoint_file(version: int, active: list[str],
                    tombstones: list[str] | None = None) -> InputFile:
    n = max(len(active), len(tombstones or []), 1)
    add_rows = [{"path": p, "size": 1, "modificationTime": None,
                 "dataChange": False} for p in active] + [None] * (n - len(active))
    rm_rows = [{"path": p, "deletionTimestamp": None,
                "dataChange": False} for p in (tombstones or [])]
    rm_rows += [None] * (n - len(rm_rows))
    table = pa.table({
        "add": pa.array(add_rows, type=ADD_TYPE),
        "remove": pa.array(rm_rows, type=pa.struct([
            ("path", pa.string()),
            ("deletionTimestamp", pa.timestamp("us")),
            ("dataChange", pa.bool_()),
        ])),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return InputFile(filename=f"{version:020d}.checkpoint.parquet",
                     data=buf.getvalue())


def manifest(paths: list[str]) -> InputFile:
    return InputFile(filename="existing_files.txt",
                     data=("\n".join(paths) + "\n").encode("utf-8"))


def f(n: int) -> str:
    return f"data/part-{n:05d}.parquet"


def basic_commits() -> list[InputFile]:
    return [
        commit_file(0, [add(f(0)), add(f(1))]),
        commit_file(1, [add(f(2))]),
        commit_file(2, [add(f(3)), remove(f(1))]),
        commit_file(3, [add(f(4)), remove(f(0))]),
    ]


# ---------- 归一化与解析 ----------

def test_normalize_data_path():
    assert normalize_data_path("./data/a.parquet") == "data/a.parquet"
    assert normalize_data_path("/data/a.parquet") == "data/a.parquet"
    assert normalize_data_path("data/space%20file.parquet") == "data/space file.parquet"
    with pytest.raises(AnalyzerError):
        normalize_data_path(123)


def test_parse_commit_handles_ndjson_and_array():
    nd = parse_commit(0, b'{"add": {"path": "a.parquet"}}\n{"remove": {"path": "b.parquet"}}',
                      "x.json")
    assert nd.adds == [("a.parquet", {"path": "a.parquet"})]
    assert nd.removes == ["b.parquet"]

    arr = parse_commit(1, b'[{"add": {"path": "c.parquet"}}]', "y.json")
    assert arr.adds[0][0] == "c.parquet"


def test_parse_checkpoint_basic():
    ck = checkpoint_file(2, [f(0), f(2)], tombstones=[f(1)])
    parsed = parse_checkpoint(2, ck.data, ck.filename)
    assert parsed.adds == [f(0), f(2)]
    assert parsed.tombstones == [f(1)]
    assert parsed.dv_detected is False


def test_bad_parquet_rejected():
    with pytest.raises(AnalyzerError):
        parse_checkpoint(1, b"not parquet", "00000000000000000001.checkpoint.parquet")


# ---------- 完整分析：仅提交 ----------

def test_replay_commits_only_all_recoverable():
    report = analyze(basic_commits(), [manifest([f(i) for i in range(5)])], 0, 3)
    statuses = {v["version"]: v["status"] for v in report["versions"]}
    assert statuses == {0: STATUS_RECOVERABLE, 1: STATUS_RECOVERABLE,
                        2: STATUS_RECOVERABLE, 3: STATUS_RECOVERABLE}
    v3 = report["versions"][3]
    assert v3["referenced_files"] == sorted([f(2), f(3), f(4)])
    assert v3["base"] == "commit-0"
    assert v3["missing_files"] == []
    assert report["summary"][STATUS_RECOVERABLE] == 4


def test_remove_then_readd_same_path():
    commits = [
        commit_file(0, [add(f(0))]),
        commit_file(1, [remove(f(0))]),
        commit_file(2, [add(f(0))]),
    ]
    report = analyze(commits, [manifest([f(0)])], 2, 2)
    v2 = report["versions"][0]
    assert v2["status"] == STATUS_RECOVERABLE
    assert v2["referenced_files"] == [f(0)]


def test_missing_data_file_flagged_with_path():
    report = analyze(basic_commits(), [manifest([f(0), f(1), f(2), f(3)])], 3, 3)
    v3 = report["versions"][0]
    assert v3["status"] == STATUS_MISSING_FILES
    assert v3["missing_files"] == [f(4)]
    assert v3["missing_commits"] == []


def test_missing_commit_breaks_replay_and_lists_version():
    commits = [c for c in basic_commits() if not c.filename.startswith(f"{2:020d}")]
    report = analyze(commits, [manifest([f(i) for i in range(5)])], 0, 3)
    by_ver = {v["version"]: v for v in report["versions"]}
    assert by_ver[0]["status"] == STATUS_RECOVERABLE
    assert by_ver[1]["status"] == STATUS_RECOVERABLE
    assert by_ver[2]["missing_commits"] == [2]
    assert by_ver[2]["status"] == STATUS_MISSING_COMMIT
    # v3 因 v2 断档同样不可重建，但仍把缺失提交号明确列出
    assert by_ver[3]["status"] == STATUS_MISSING_COMMIT
    assert by_ver[3]["missing_commits"] == [2]


def test_no_base_without_commit_zero_or_checkpoint():
    commits = [commit_file(1, [add(f(0))])]
    report = analyze(commits, [manifest([f(0)])], 0, 1)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_MISSING_COMMIT
    assert v0["missing_commits"] == [0]
    assert v0["base"] is None


# ---------- 完整分析：检查点基线 ----------

def test_checkpoint_base_short_circuits_missing_early_commits():
    # 只有 v2 检查点 + v3 提交；v0/v1 提交缺失，但检查点不晚于目标版本
    logs = [checkpoint_file(2, [f(0), f(2), f(3)]),
            commit_file(3, [add(f(4)), remove(f(0))])]
    report = analyze(logs, [manifest([f(0), f(2), f(3), f(4)])], 2, 3)
    v2, v3 = report["versions"]
    assert v2["status"] == STATUS_RECOVERABLE
    assert v2["base"] == "checkpoint-2"
    assert v3["base"] == "checkpoint-2"
    assert v3["referenced_files"] == sorted([f(2), f(3), f(4)])


def test_checkpoint_tombstones_not_part_of_snapshot():
    ck = checkpoint_file(1, [f(0)], tombstones=[f(9)])
    report = analyze([ck], [manifest([f(0)])], 1, 1)
    v1 = report["versions"][0]
    assert v1["referenced_files"] == [f(0)]
    assert f(9) not in v1["referenced_files"]


def test_checkpoint_older_than_target_requires_continuous_commits():
    logs = [checkpoint_file(1, [f(0)]),
            commit_file(2, [add(f(1))])]
    # 检查点之后的 v2 提交存在，连续重放成功
    ok = analyze(logs, [manifest([f(0), f(1)])], 2, 2)
    assert ok["versions"][0]["status"] == STATUS_RECOVERABLE

    # v2 提交缺失 -> 检查点基线之后断档
    bad = analyze([checkpoint_file(1, [f(0)])], [manifest([f(0)])], 2, 2)
    v = bad["versions"][0]
    assert v["status"] == STATUS_MISSING_COMMIT
    assert v["missing_commits"] == [2]


# ---------- 删除向量拒绝 ----------

def test_deletion_vector_in_commit_rejected():
    dv = {"storageType": "u", "pathOrInlineDv": "abc"}
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([commit_file(0, [add(f(0), dv)])], [manifest([])], 0, 0)
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([commit_file(0, [remove(f(0), dv)])], [manifest([])], 0, 0)


def test_deletion_vector_protocol_feature_rejected():
    commits = [commit_file(0, [
        {"protocol": {"minReaderVersion": 3, "minWriterVersion": 7,
                      "readerFeatures": ["deletionVectors"],
                      "writerFeatures": ["deletionVectors"]}},
        add(f(0)),
    ])]
    with pytest.raises(AnalyzerError, match="deletionVectors"):
        analyze(commits, [manifest([f(0)])], 0, 0)


def test_deletion_vector_in_checkpoint_rejected():
    table = pa.table({"add": pa.array([
        {"path": f(0), "deletionVector": {"storageType": "u",
                                          "pathOrInlineDv": "x"}},
    ], type=ADD_DV_TYPE)})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    ck = InputFile(filename=f"{1:020d}.checkpoint.parquet", data=buf.getvalue())
    with pytest.raises(AnalyzerError, match="删除向量"):
        analyze([ck], [manifest([f(0)])], 1, 1)


# ---------- 输入校验与杂项 ----------

def test_range_validation():
    with pytest.raises(AnalyzerError, match="起点不能大于终点"):
        analyze(basic_commits(), [], 3, 1)
    with pytest.raises(AnalyzerError, match="负数"):
        analyze(basic_commits(), [], -1, 1)


def test_empty_and_unrecognized_inputs():
    with pytest.raises(AnalyzerError, match="至少上传"):
        analyze([], [], 0, 1)
    weird = InputFile(filename="README.md", data=b"hi")
    with pytest.raises(AnalyzerError, match="没有可识别"):
        analyze([weird], [], 0, 0)


def test_ignored_multi_part_checkpoint_reported():
    logs = basic_commits() + [InputFile(
        filename=f"{1:020d}.checkpoint.00000000000000000001.00000000000000000002.parquet",
        data=b"x")]
    report = analyze(logs, [manifest([f(i) for i in range(5)])], 0, 0)
    ignored_names = [i["name"] for i in report["ignored_files"]]
    assert any("checkpoint" in n and n.endswith(".parquet") for n in ignored_names)


def test_manifest_comments_blank_lines_and_duplicates():
    m = InputFile(filename="m.txt",
                  data=b"# comment\n\n./data/part-00000.parquet\n/data/part-00000.parquet\n")
    report = analyze([commit_file(0, [add("data/part-00000.parquet")])], [m], 0, 0)
    assert report["existing_file_count"] == 1
    assert report["versions"][0]["status"] == STATUS_RECOVERABLE


def test_duplicate_upload_deduped():
    c = commit_file(0, [add(f(0))])
    report = analyze([c, InputFile(filename=c.filename, data=c.data)],
                     [manifest([f(0)])], 0, 0)
    assert c.filename in report["duplicate_files"]
    assert report["versions"][0]["status"] == STATUS_RECOVERABLE


def test_no_manifest_means_every_referenced_file_is_missing():
    report = analyze(basic_commits(), [], 2, 2)
    v2 = report["versions"][0]
    assert v2["status"] == STATUS_MISSING_FILES
    assert v2["missing_files"] == sorted([f(0), f(2), f(3)])
    assert report["existing_file_count"] == 0


# ---------- 清单路径格式：原样 / URI 编码 / 未注明 ----------

SPACE = "data/part-00000-literal name.parquet"
LITERAL = "data/part-00000-literal%20name.parquet"


def uri_log_path(path: str) -> str:
    """模拟 Delta 日志里的 URI 编码路径（字面 %20 会变成 %2520）。"""
    import urllib.parse
    return urllib.parse.quote(path, safe="/")


def test_decode_uri_path_once_only():
    from app.analyzer import decode_uri_path_once, normalize_raw_path
    # 只解码一次：%2520 变成字面 %20，而不是空格
    assert decode_uri_path_once("data/a%2520b.parquet") == "data/a%20b.parquet"
    assert decode_uri_path_once("./data/a%20b.parquet") == "data/a b.parquet"
    # 原样入口不解码
    assert normalize_raw_path("data/a%20b.parquet") == "data/a%20b.parquet"
    assert normalize_raw_path("./data/a%20b.parquet") == "data/a%20b.parquet"
    assert normalize_raw_path("/data/a%20b.parquet") == "data/a%20b.parquet"


def _encoding_logs() -> list[InputFile]:
    # 日志引用：字面 %20 文件与含空格文件各一个
    return [commit_file(0, [add(uri_log_path(LITERAL)),
                            add(uri_log_path(SPACE))])]


def test_raw_manifest_keeps_literal_percent20_and_space_distinct():
    report = analyze(_encoding_logs(),
                     [InputFile(filename="export.raw.txt",
                                data=f"{LITERAL}\n{SPACE}\n".encode(),
                                path_format="raw")], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_RECOVERABLE
    assert v0["missing_files"] == []
    assert report["existing_file_count"] == 2


def test_uri_manifest_decodes_exactly_once():
    # URI 清单里字面 %20 编码为 %2520、空格编码为 %20，各解码一次
    body = f"{uri_log_path(LITERAL)}\n{uri_log_path(SPACE)}\n".encode()
    report = analyze(_encoding_logs(),
                     [InputFile(filename="backup.uri.txt", data=body,
                                path_format="uri")], 0, 0)
    assert report["versions"][0]["status"] == STATUS_RECOVERABLE


def test_raw_manifest_space_encoded_line_does_not_confirm_space_file():
    # 原样清单把两个文件名都写成 %20 —— 只能确认字面 %20 的那个文件，
    # 含空格的那个文件（日志里是 %20）仍缺失，不能被误判为可回退
    body = f"{LITERAL}\ndata/part-00000-space%20name.parquet\n".encode()
    report = analyze(_encoding_logs(),
                     [InputFile(filename="export.raw.txt", data=body,
                                path_format="raw")], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_MISSING_FILES
    assert v0["missing_files"] == [SPACE]
    assert LITERAL not in v0["missing_files"]


def test_unmarked_legacy_entry_same_raw_and_decoded_confirms():
    report = analyze([commit_file(0, [add("data/plain.parquet")])],
                     [InputFile(filename="legacy.txt",
                                data=b"data/plain.parquet\n")], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_RECOVERABLE
    assert report["manifests"] == ["legacy.txt"]


def test_unmarked_legacy_ambiguous_entry_marks_both_paths_pending():
    # 旧清单只写 %20，原样和 URI 解码结果不同：两条路径都待确认，
    # 既不能算现存，也不能判为缺失
    report = analyze(_encoding_logs(),
                     [InputFile(filename="legacy.txt",
                                data=f"{LITERAL}\n".encode())], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_PENDING_CONFIRMATION
    assert v0["missing_files"] == []
    assert LITERAL in v0["pending_files"]
    assert SPACE in v0["pending_files"]
    assert report["summary"][STATUS_PENDING_CONFIRMATION] == 1
    for detail in v0["file_details"]:
        if detail["path"] in (LITERAL, SPACE):
            assert detail["state"] == "pending"
            assert detail["evidence"][0]["source"] == "legacy.txt"


def test_explicit_entry_overrides_ambiguous_legacy_entry():
    # 未注明格式的旧行把字面 %20 与空格两条路径都标为待确认；
    # 另一份原样清单明确证实字面 %20 文件存在，则它仍算现存，
    # 只剩空格文件待确认 -> 版本判为待确认而不是缺失
    logs = _encoding_logs()
    legacy = InputFile(filename="legacy.txt",
                       data=f"{LITERAL}\n".encode())
    explicit = InputFile(filename="export.raw.txt",
                         data=f"{LITERAL}\n".encode(), path_format="raw")
    report = analyze(logs, [legacy, explicit], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_PENDING_CONFIRMATION
    assert LITERAL not in v0["pending_files"]
    assert SPACE in v0["pending_files"]
    detail = {d["path"]: d["state"] for d in v0["file_details"]}
    assert detail[LITERAL] == "confirmed"
    assert detail[SPACE] == "pending"
    assert report["existing_file_count"] == 1


def test_missing_commit_takes_precedence_over_pending():
    logs = [commit_file(0, [add("data/plain%20name.parquet")])]
    report = analyze(logs, [InputFile(filename="legacy.txt",
                                      data=b"data/plain%20name.parquet\n")],
                     1, 1)
    v1 = report["versions"][0]
    assert v1["status"] == STATUS_MISSING_COMMIT
    assert v1["missing_commits"] == [1]


def test_manifest_format_directive_and_filename_hint():
    from app.analyzer import parse_manifest_entries, PATH_FORMAT_RAW, PATH_FORMAT_URI
    entries, fmt = parse_manifest_entries(
        b"# format: raw\ndata/a%20b.parquet\n", "m.txt")
    assert fmt == PATH_FORMAT_RAW
    assert entries[0].raw == "data/a%20b.parquet"
    entries, fmt = parse_manifest_entries(
        b"data/a%20b.parquet\n", "object_listing.raw.txt")
    assert fmt == PATH_FORMAT_RAW
    entries, fmt = parse_manifest_entries(
        b"data/a%20b.parquet\n", "backup.uri.txt")
    assert fmt == PATH_FORMAT_URI
    entries, fmt = parse_manifest_entries(
        "# 格式：URI 编码\ndata/a%2520b.parquet\n".encode(), "m.txt")
    assert fmt == PATH_FORMAT_URI


def test_manifest_format_declared_in_input_file():
    assert InputFile(filename="m", data=b"", format="uri").path_format == "uri"
    assert InputFile(filename="m", data=b"", path_format="raw").path_format == "raw"
    with pytest.raises(AnalyzerError, match="路径格式"):
        analyze([commit_file(0, [add("a.parquet")])],
                [InputFile(filename="m", data=b"a.parquet\n",
                           path_format="bogus")], 0, 0)


def test_json_commit_and_checkpoint_paths_decode_once():
    lit_log = uri_log_path(LITERAL)  # data/...literal%2520name...
    commit = parse_commit(0, json.dumps({"add": {"path": lit_log}}).encode(),
                          "0.json")
    assert commit.adds[0][0] == LITERAL
    ck = checkpoint_file(0, [lit_log])
    parsed = parse_checkpoint(0, ck.data, ck.filename)
    assert parsed.adds == [LITERAL]


def test_explicit_missing_file_wins_over_pending():
    # 一条引用文件被明确判定缺失，另一条待确认：版本仍应报缺失文件
    report = analyze([commit_file(0, [add(uri_log_path(LITERAL)),
                                      add("data/gone.parquet")])],
                     [InputFile(filename="legacy.txt",
                                data=f"{LITERAL}\n".encode())], 0, 0)
    v0 = report["versions"][0]
    assert v0["status"] == STATUS_MISSING_FILES
    assert v0["missing_files"] == ["data/gone.parquet"]
    assert LITERAL in v0["pending_files"]


def test_same_scenario_three_manifest_modes():
    """同一行 ``…%20…`` 在三种清单格式下的结论必须不同。

    日志引用的是含空格文件（URI 编码为 ``literal%20name``）：
    - 原样清单：该行指的是字面 %20 文件，含空格文件缺失；
    - URI 清单：解码一次得到含空格文件，可回退；
    - 未注明清单：两种解释都成立，两条路径待确认。
    """
    import urllib.parse
    literal = "data/part-00002-literal%20name.parquet"
    space = "data/part-00002-literal name.parquet"
    logs = [commit_file(0, [add(urllib.parse.quote(space, safe="/"))])]
    body = f"{literal}\n".encode()

    raw = analyze(logs, [InputFile(filename="a.raw.txt", data=body,
                                   path_format="raw")], 0, 0)
    assert raw["versions"][0]["status"] == STATUS_MISSING_FILES
    assert raw["versions"][0]["missing_files"] == [space]

    uri = analyze(logs, [InputFile(filename="a.uri.txt", data=body,
                                   path_format="uri")], 0, 0)
    assert uri["versions"][0]["status"] == STATUS_RECOVERABLE

    auto = analyze(logs, [InputFile(filename="legacy.txt", data=body)], 0, 0)
    assert auto["versions"][0]["status"] == STATUS_PENDING_CONFIRMATION
    assert auto["versions"][0]["pending_files"] == [space]
    assert auto["pending_file_count"] == 2


def test_analyze_accepts_manifest_format_arguments():
    import urllib.parse
    literal = "data/lit%20name.parquet"
    logs = [commit_file(0, [add(urllib.parse.quote(literal, safe="/"))])]
    mf = [InputFile(filename="m.txt", data=f"{literal}\n".encode())]
    # 未声明 -> 歧义待确认
    assert analyze(logs, mf, 0, 0)["versions"][0]["status"] == \
        STATUS_PENDING_CONFIRMATION
    # 全局格式
    assert analyze(logs, mf, 0, 0,
                   manifest_format="raw")["versions"][0]["status"] == \
        STATUS_RECOVERABLE
    # 按位置列表
    assert analyze(logs, mf, 0, 0,
                   manifest_formats=["raw"])["versions"][0]["status"] == \
        STATUS_RECOVERABLE
    # 按位置字典
    assert analyze(logs, mf, 0, 0,
                   manifest_formats={0: "uri"})["versions"][0]["status"] == \
        STATUS_MISSING_FILES
    with pytest.raises(AnalyzerError, match="列表或字典"):
        analyze(logs, mf, 0, 0, manifest_formats="raw")
