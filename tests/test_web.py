"""Flask 路由层测试（含本地样例加载）。"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from app.web import app


def _commit_bytes(actions):
    return ("\n".join(json.dumps(a) for a in actions)).encode("utf-8")


def test_index_renders():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Delta Lake 版本可恢复性分析".encode() in resp.data


def test_upload_analyze_flow():
    client = app.test_client()
    data = {
        "from_version": "0",
        "to_version": "1",
        "log_files": [
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/a.parquet"}}])),
             "00000000000000000000.json"),
            (io.BytesIO(_commit_bytes([{"add": {"path": "data/b.parquet"}}])),
             "00000000000000000001.json"),
        ],
        "manifest_files": [
            (io.BytesIO(b"data/a.parquet\ndata/b.parquet\n"), "existing.txt"),
        ],
    }
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "可重建 · 文件齐全".encode() in resp.data


def test_upload_missing_file_reports_400():
    client = app.test_client()
    resp = client.post("/analyze", data={
        "from_version": "0", "to_version": "1",
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "至少上传".encode() in resp.data


def test_sample_api_healthy_and_incident():
    client = app.test_client()
    for name, expected_recoverable in (("healthy_table", 5),
                                       ("incident_table", None)):
        resp = client.post(f"/api/analyze-sample/{name}")
        assert resp.status_code == 200, resp.data
        report = resp.get_json()
        assert report["range"] == {"from": 0, "to": (4 if name == "healthy_table" else 5)}
        if expected_recoverable:
            assert report["summary"]["recoverable"] == expected_recoverable


def test_unknown_sample_404():
    resp = app.test_client().post("/api/analyze-sample/nope")
    assert resp.status_code == 404


def test_sample_api_list():
    resp = app.test_client().get("/api/samples")
    assert resp.status_code == 200
    names = resp.get_json()
    assert "healthy_table" in names and "incident_table" in names


def _commit(version: int, path: str):
    return (io.BytesIO((json.dumps({"add": {"path": path}}) + "\n").encode()),
            f"{version:020d}.json")


def test_upload_per_manifest_formats():
    import urllib.parse
    literal = "data/part-00000-literal%20name.parquet"
    space = "data/part-00000-literal name.parquet"
    log_paths = [urllib.parse.quote(literal, safe="/"),
                 urllib.parse.quote(space, safe="/")]
    data = {
        "from_version": "0", "to_version": "0",
        "log_files": [_commit(0, log_paths[0]), _commit(0, log_paths[1])],
        "manifest_files": [
            (io.BytesIO(f"{literal}\n".encode()),
             "object_store_export.raw.txt"),
            (io.BytesIO(f"{urllib.parse.quote(literal, safe='/')}\n"
                        f"{urllib.parse.quote(space, safe='/')}\n".encode()),
             "backup_manifest.uri.txt"),
        ],
        "manifest_format_0": "raw",
        "manifest_format_1": "uri",
    }
    client = app.test_client()
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "可重建 · 文件齐全".encode() in resp.data


def test_upload_global_manifest_format_and_json_list():
    literal = "data/part-00000-literal%20name.parquet"
    import urllib.parse
    log_path = urllib.parse.quote(literal, safe="/")
    base = {
        "from_version": "0", "to_version": "0",
        "log_files": [_commit(0, log_path)],
        "manifest_files": [(io.BytesIO(f"{literal}\n".encode()), "m.txt")],
    }
    client = app.test_client()
    for extra in ({"manifest_path_format": "raw"},
                  {"manifest_formats": '["raw"]'},
                  {"manifest_formats": "raw"}):
        data = dict(base, **extra)
        # 每次请求都要用新的 BytesIO，上传流在第一次请求后就被关闭
        data["log_files"] = [_commit(0, log_path)]
        data["manifest_files"] = [
            (io.BytesIO(f"{literal}\n".encode()), "m.txt")]
        resp = client.post("/analyze", data=data,
                           content_type="multipart/form-data")
        assert resp.status_code == 200, resp.data
        assert "可重建 · 文件齐全".encode() in resp.data


def test_upload_unmarked_legacy_manifest_shows_pending():
    literal = "data/part-00000-literal%20name.parquet"
    import urllib.parse
    log_path = urllib.parse.quote(literal, safe="/")
    data = {
        "from_version": "0", "to_version": "0",
        "log_files": [_commit(0, log_path)],
        "manifest_files": [(io.BytesIO(f"{literal}\n".encode()),
                            "legacy_listing.txt")],
    }
    resp = app.test_client().post("/analyze", data=data,
                                  content_type="multipart/form-data")
    assert resp.status_code == 200
    assert "待确认".encode() in resp.data


def test_sample_api_encoding_and_legacy():
    client = app.test_client()
    enc = client.post("/api/analyze-sample/encoding_table").get_json()
    assert enc["range"] == {"from": 0, "to": 2}
    by_ver = {v["version"]: v for v in enc["versions"]}
    # 字面 %20 与空格两个文件都被明确清单证实
    assert by_ver[1]["status"] == "recoverable"
    # 只差 %20/空格 的第三个文件被误删 -> 缺失，而不是被旧逻辑当成可回退
    assert by_ver[2]["status"] == "missing_files"
    assert by_ver[2]["missing_files"] == [
        "data/part-00002-literal name.parquet"]

    legacy = client.post("/api/analyze-sample/legacy_manifest_table").get_json()
    by_ver = {v["version"]: v for v in legacy["versions"]}
    assert by_ver[0]["status"] == "pending_confirmation"
    # 未注明格式的编码行：原样（字面 %20）与解码后（空格）两条路径都待确认；
    # 逐版只列出快照实际引用到的待确认文件
    assert by_ver[0]["pending_files"] == [
        "data/part-00000-literal%20name.parquet"]
    evidence = by_ver[0]["file_details"][0]["evidence"][0]
    assert evidence["raw"] == "data/part-00000-literal%20name.parquet"
    assert evidence["decoded"] == "data/part-00000-literal name.parquet"
    assert legacy["pending_file_count"] == 2
    assert legacy["manifest_info"][0]["path_format"] == "auto"


def test_sample_list_includes_new_samples():
    names = app.test_client().get("/api/samples").get_json()
    assert {"encoding_table", "legacy_manifest_table"} <= set(names)
