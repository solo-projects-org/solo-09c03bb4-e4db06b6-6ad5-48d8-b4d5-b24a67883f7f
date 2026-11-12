"""生成两套本地样例 Delta 表，供网页一键加载。

样例只构造 _delta_log 中的 JSON 提交、经典单文件 Parquet 检查点
和现存数据文件路径清单，不依赖真实 Spark / Delta Lake 运行时。

运行：python -I scripts/generate_samples.py
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# ---- Delta 检查点列结构（仅保留重放所需字段，无 deletionVector 列） ----
ADD_TYPE = pa.struct([
    ("path", pa.string()),
    ("size", pa.int64()),
    ("modificationTime", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
REMOVE_TYPE = pa.struct([
    ("path", pa.string()),
    ("deletionTimestamp", pa.timestamp("us")),
    ("dataChange", pa.bool_()),
])
PROTOCOL_TYPE = pa.struct([
    ("minReaderVersion", pa.int32()),
    ("minWriterVersion", pa.int32()),
    ("readerFeatures", pa.list_(pa.string())),
    ("writerFeatures", pa.list_(pa.string())),
])


def commit_name(version: int) -> str:
    return f"{version:020d}.json"


def checkpoint_name(version: int) -> str:
    return f"{version:020d}.checkpoint.parquet"


def write_commit(log_dir: Path, version: int, actions: list[dict]) -> None:
    lines = [json.dumps(action, ensure_ascii=False) for action in actions]
    (log_dir / commit_name(version)).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def add_action(path: str) -> dict:
    return {"add": {"path": path, "partitionValues": {}, "size": 1024,
                    "modificationTime": 1700000000000, "dataChange": True}}


def remove_action(path: str) -> dict:
    return {"remove": {"path": path, "deletionTimestamp": 1700000100000,
                       "dataChange": True}}


PROTOCOL_ACTION = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}


def write_checkpoint(log_dir: Path, version: int,
                     active_paths: list[str], tombstone_paths: list[str]) -> None:
    """写经典单文件 Parquet 检查点。"""
    n_add, n_rm = len(active_paths), len(tombstone_paths)
    n_rows = max(n_add, n_rm, 1)  # protocol 至少占一行

    add_rows = [
        {"path": p, "size": 1024,
         "modificationTime": 1700000000000, "dataChange": False}
        for p in active_paths
    ] + [None] * (n_rows - n_add)
    remove_rows = [
        {"path": p, "deletionTimestamp": 1700000100000, "dataChange": False}
        for p in tombstone_paths
    ] + [None] * (n_rows - n_rm)
    protocol_rows = [
        {"minReaderVersion": 1, "minWriterVersion": 2,
         "readerFeatures": [], "writerFeatures": []}
    ] + [None] * (n_rows - 1)

    table = pa.table({
        "add": pa.array(add_rows, type=ADD_TYPE),
        "remove": pa.array(remove_rows, type=REMOVE_TYPE),
        "protocol": pa.array(protocol_rows, type=PROTOCOL_TYPE),
    })
    pq.write_table(table, log_dir / checkpoint_name(version))


def write_manifest(table_dir: Path, existing: list[str]) -> None:
    lines = ["# 现存数据文件表内相对路径清单（# 开头为注释，空行忽略）"]
    for i, p in enumerate(existing):
        # 故意混入 ./ 前缀与空行，验证服务端归一化逻辑
        lines.append(("./" + p) if i % 2 == 0 else p)
        if i == 1:
            lines.append("")
    (table_dir / "existing_files.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    # 同步建占位文件，让 samples 目录看起来像真实表布局
    for p in existing:
        target = table_dir / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder parquet bytes for sample")


def write_raw_manifest(path: Path, existing: list[str], note: str) -> None:
    """写对象存储导出的原样路径清单（不做 URI 编码）。"""
    lines = [f"# {note}",
             "# format: raw  —— 原样路径，服务端不解码"]
    lines.extend(existing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_uri_manifest(path: Path, existing: list[str], note: str) -> None:
    """写备份清单，路径统一 URI 编码（空格等写成 %XX）。"""
    lines = [f"# {note}",
             "# format: uri  —— URI 编码路径，服务端只解码一次"]
    lines.extend(urllib.parse.quote(p, safe="/") for p in existing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def build_healthy() -> None:
    """健康表：v0-v4 全部可重建，含 v2 检查点，所有数据文件均在。"""
    table_dir = SAMPLES / "healthy_table"
    reset_dir(table_dir)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True)

    f = lambda n: f"data/part-{n:05d}-healthy.parquet"
    write_commit(log_dir, 0, [PROTOCOL_ACTION, add_action(f(0)), add_action(f(1))])
    write_commit(log_dir, 1, [add_action(f(2))])
    write_commit(log_dir, 2, [add_action(f(3)), remove_action(f(1))])
    # v2 检查点：当前活动文件 f0,f2,f3；f1 是墓碑
    write_checkpoint(log_dir, 2, [f(0), f(2), f(3)], [f(1)])
    write_commit(log_dir, 3, [add_action(f(4))])
    write_commit(log_dir, 4, [add_action(f(5)), remove_action(f(2))])

    (log_dir / "_last_checkpoint").write_text(json.dumps({
        "version": 2, "size": 0}), encoding="utf-8")
    write_manifest(table_dir, [f(i) for i in range(6)])


def build_incident() -> None:
    """误删事故表：v3 提交丢失、a4/a6 数据文件被删，但 v2/v4 检查点留存。"""
    table_dir = SAMPLES / "incident_table"
    reset_dir(table_dir)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True)

    a = lambda n: f"data/part-{n:05d}-incident.parquet"
    write_commit(log_dir, 0, [PROTOCOL_ACTION, add_action(a(0)), add_action(a(1))])
    write_commit(log_dir, 1, [add_action(a(2))])
    write_commit(log_dir, 2, [add_action(a(3)), remove_action(a(1))])
    write_checkpoint(log_dir, 2, [a(0), a(2), a(3)], [a(1)])

    # 事故中 v3 提交（add a4）被误删，且没有留下备份
    v3_lost = log_dir / commit_name(3)
    assert not v3_lost.exists()

    write_commit(log_dir, 4, [add_action(a(5))])
    # v4 检查点在 v3 提交丢失前生成，保留了 v3 引入的 a4
    write_checkpoint(log_dir, 4, [a(0), a(2), a(3), a(4), a(5)], [a(1)])
    write_commit(log_dir, 5, [add_action(a(6)), remove_action(a(2))])

    # 一个分片检查点残片：本工具明确不支持，应在页面中列为“已忽略”
    (log_dir / f"{1:020d}.checkpoint.00000000000000000001.00000000000000000002.parquet")\
        .write_bytes(b"not-a-real-multi-part-checkpoint")
    (log_dir / "_last_checkpoint").write_text(json.dumps({
        "version": 4, "size": 0}), encoding="utf-8")

    # 现存清单：a4、a6 已随误删消失；a1 虽已逻辑删除但物理文件仍在
    write_manifest(table_dir, [a(0), a(1), a(2), a(3), a(5)])


def build_encoding() -> None:
    """编码核对表：演示含空格与含字面 %20 的文件如何分别判断。

    - ``part-00001-space name.parquet``：日志里存 ``…space%20name…``，
      按规范解码一次得到空格路径，该文件现存；
    - ``part-00002-literal%20name.parquet``：文件名里本来就是字面 ``%20``，
      日志里存 ``…literal%2520name…``，解码一次仍是字面 ``%20``，该文件现存；
    - ``part-00002-literal name.parquet``（含空格）已被误删，日志里存
      ``…literal%20name…``。原样清单里另有一个字面 ``%20`` 的同名文件；
      如果照旧把清单里的 ``%20`` 当成空格，就会把这条已删除的空格文件
      误判为现存 —— 本样例用它验证修复后判为缺失。
    """
    table_dir = SAMPLES / "encoding_table"
    reset_dir(table_dir)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True)

    literal0 = "data/part-00000-literal%20name.parquet"
    space1 = "data/part-00001-space name.parquet"
    literal2 = "data/part-00002-literal%20name.parquet"
    # 与 literal2 只差在 %20 / 空格：这条文件已被误删，且未出现在任何清单
    space2 = "data/part-00002-literal name.parquet"

    def log_path(p: str) -> str:
        # Delta 日志中的路径按规范 URI 编码：字面 %20 会变成 %2520
        return urllib.parse.quote(p, safe="/")

    write_commit(log_dir, 0,
                 [PROTOCOL_ACTION, add_action(log_path(literal0))])
    write_commit(log_dir, 1, [add_action(log_path(space1))])
    write_commit(log_dir, 2, [add_action(log_path(literal2)),
                              add_action(log_path(space2))])

    # 对象存储导出的原样清单：文件名按字面写出，%20 就是文件名的一部分
    write_raw_manifest(
        table_dir / "object_store_export.raw.txt",
        [literal0, space1, literal2],
        "对象存储导出（原样文件名）：空格就是空格，%20 就是字面 %20")
    # 备份清单：URI 编码，解码一次后分别得到空格与字面 %20
    write_uri_manifest(
        table_dir / "backup_manifest.uri.txt",
        [literal0, space1, literal2],
        "备份清单（URI 编码）：%20 与 %2520 各解码一次")

    (log_dir / "_last_checkpoint").write_text(json.dumps({
        "version": 2, "size": 0}), encoding="utf-8")

    for p in (literal0, space1, literal2):
        target = table_dir / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder parquet bytes for sample")
    # space2 故意不创建、也不写进清单：它就是在事故中被删掉的那个文件


def build_legacy_manifest() -> None:
    """旧清单表：清单未注明格式，原样与解码结果不同的条目都待确认。

    旧清单里两行：``part-00000-literal%20name.parquet``（若按原样是字面
    %20，若按 URI 是空格）与 ``part-00001-space name.parquet``（原样即空格，
    解码后不变）。没有第二个来源佐证时，前者两条解释指向不同路径，标为
    待确认；后者原样与解码相同，仍算现存。
    """
    table_dir = SAMPLES / "legacy_manifest_table"
    reset_dir(table_dir)
    log_dir = table_dir / "_delta_log"
    log_dir.mkdir(parents=True)

    literal0 = "data/part-00000-literal%20name.parquet"
    space1 = "data/part-00001-space name.parquet"

    def log_path(p: str) -> str:
        return urllib.parse.quote(p, safe="/")

    write_commit(log_dir, 0,
                 [PROTOCOL_ACTION, add_action(log_path(literal0))])
    write_commit(log_dir, 1, [add_action(log_path(space1))])

    # 旧清单未注明格式：既不写 # format:，文件名也不带 raw/uri 提示。
    # 这里是对象存储导出的原样清单，但服务端不能假定，需人工确认。
    (table_dir / "legacy_listing.txt").write_text(
        "# 旧的现存文件清单（未注明路径格式）\n"
        f"{literal0}\n"
        f"{space1}\n",
        encoding="utf-8")

    for p in (literal0, space1):
        target = table_dir / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder parquet bytes for sample")


def main() -> None:
    build_healthy()
    build_incident()
    build_encoding()
    build_legacy_manifest()
    print(f"样例已生成于 {SAMPLES}")
    for table in sorted(SAMPLES.iterdir()):
        if table.is_dir():
            logs = sorted(p.name for p in (table / "_delta_log").iterdir())
            print(f"\n[{table.name}] _delta_log:")
            for name in logs:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
