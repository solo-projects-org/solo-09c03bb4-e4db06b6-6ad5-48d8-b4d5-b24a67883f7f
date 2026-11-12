"""Flask 网页：Delta Lake 版本可恢复性分析。

值班员在页面上填写版本区间，上传 _delta_log 留存的 JSON 提交、
经典单文件 Parquet 检查点，以及现存数据文件的表内相对路径清单，
提交后逐版展示可恢复性结论。页面也可一键加载仓库内置的本地样例。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from .analyzer import AnalyzerError, InputFile, analyze

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 单次请求上限 256MB

SAMPLE_ROOT = Path(os.environ.get("SAMPLE_ROOT",
                                  Path(__file__).resolve().parent.parent / "samples"))

# 样例目录名 -> 页面展示名 / 默认版本区间 / 默认加载的清单文件
SAMPLES = {
    "healthy_table": {
        "label": "健康表样例（0-4 版均可重建）",
        "range": (0, 4),
        "manifests": ["existing_files.txt"],
    },
    "incident_table": {
        "label": "误删事故样例（部分版本缺提交或缺数据文件）",
        "range": (0, 5),
        "manifests": ["existing_files.txt"],
    },
    "encoding_table": {
        "label": "编码核对样例（空格与字面 %20 分别判为现存/缺失）",
        "range": (0, 2),
        "manifests": ["object_store_export.raw.txt", "backup_manifest.uri.txt"],
    },
    "legacy_manifest_table": {
        "label": "旧清单样例（未注明格式，编码行标为待确认）",
        "range": (0, 1),
        "manifests": ["legacy_listing.txt"],
    },
}

_MANIFEST_FORMAT_FIELDS = (
    "manifest_path_format",
    "manifest_format",
    "path_format",
)


def _read_uploads(field_name: str) -> list[InputFile]:
    files = []
    for storage in request.files.getlist(field_name):
        if not storage or storage.filename == "":
            continue
        files.append(InputFile(filename=storage.filename,
                               data=storage.read()))
    return files


def _manifest_formats_from_request(count: int) -> tuple[Any, str | None]:
    """收集每份清单的格式声明。

    支持三种表单写法：
    1. 逐份字段 ``manifest_format_0`` / ``manifest_format_1`` …，返回列表；
    2. ``manifest_formats``：JSON 数组（可含 ``{"path_format": ...}`` 对象）
       或逗号分隔列表，按上传顺序对应；
    3. 全局字段 ``manifest_path_format`` / ``manifest_format`` / ``path_format``，
    三者都为空时返回 ``None``，交给清单首部指令或文件名推断。
    """
    per_file = [(request.form.get(f"manifest_format_{i}") or "").strip()
                for i in range(count)]
    if any(value for value in per_file):
        return per_file, None

    raw_list = (request.form.get("manifest_formats") or "").strip()
    if raw_list:
        values: list[Any] = []
        if raw_list.startswith("["):
            try:
                parsed = json.loads(raw_list)
            except json.JSONDecodeError as exc:
                raise AnalyzerError(f"manifest_formats 不是合法 JSON：{exc}") from exc
            if not isinstance(parsed, list):
                raise AnalyzerError("manifest_formats 必须是 JSON 数组")
            values = parsed
        else:
            values = [item.strip() for item in raw_list.split(",")]
        if len(values) == 1 and count > 1:
            # 只填了一个格式时视为对所有清单生效
            values = values * count
        normalized: list[Any] = []
        for i, value in enumerate(values):
            if isinstance(value, dict):
                value = value.get("path_format") or value.get("format")
            if value is None:
                normalized.append(None)
            elif isinstance(value, str):
                normalized.append(value.strip() or None)
            else:
                raise AnalyzerError(
                    f"manifest_formats 第 {i + 1} 项必须是字符串或对象")
        return normalized, None

    for field in _MANIFEST_FORMAT_FIELDS:
        global_format = (request.form.get(field) or "").strip()
        if global_format:
            return None, global_format
    return None, None


def _parse_version(raw: str, field_label: str) -> int:
    raw = (raw or "").strip()
    if raw == "":
        raise AnalyzerError(f"请填写{field_label}")
    try:
        return int(raw)
    except ValueError as exc:
        raise AnalyzerError(f"{field_label}必须是整数，收到：{raw!r}") from exc


def _run_analysis(log_files, manifest_files, from_version, to_version,
                  manifest_formats=None, manifest_format=None):
    return analyze(log_files=log_files,
                   manifest_files=manifest_files,
                   from_version=from_version,
                   to_version=to_version,
                   manifest_formats=manifest_formats,
                   manifest_format=manifest_format)


@app.get("/")
def index():
    return render_template("index.html", samples=SAMPLES)


@app.post("/analyze")
def analyze_upload():
    try:
        from_version = _parse_version(request.form.get("from_version"), "起始版本")
        to_version = _parse_version(request.form.get("to_version"), "结束版本")
        log_files = _read_uploads("log_files")
        manifest_storages = [s for s in request.files.getlist("manifest_files")
                             if s and s.filename != ""]
        manifest_files = _read_uploads("manifest_files")
        manifest_formats, global_format = _manifest_formats_from_request(
            len(manifest_storages))
        report = _run_analysis(log_files, manifest_files,
                               from_version, to_version,
                               manifest_formats=manifest_formats,
                               manifest_format=global_format)
    except AnalyzerError as exc:
        return render_template("index.html", samples=SAMPLES, error=str(exc),
                               form=request.form), 400
    return render_template("index.html", samples=SAMPLES, report=report)


@app.get("/api/samples")
def list_samples():
    return jsonify({
        key: {"label": meta["label"], "range": list(meta["range"]),
              "manifests": list(meta.get("manifests", []))}
        for key, meta in SAMPLES.items()
    })


@app.post("/api/analyze-sample/<sample_name>")
def analyze_sample(sample_name: str):
    """直接加载服务器本地样例目录，无需手动上传。"""
    if sample_name not in SAMPLES:
        return jsonify({"error": f"未知样例：{sample_name}"}), 404
    sample_dir = SAMPLE_ROOT / sample_name
    if not sample_dir.is_dir():
        return jsonify({"error": f"样例目录不存在：{sample_dir}"}), 404

    try:
        payload = request.get_json(silent=True) or {}
        meta = SAMPLES[sample_name]
        from_version = int(payload.get("from_version", meta["range"][0]))
        to_version = int(payload.get("to_version", meta["range"][1]))

        log_dir = sample_dir / "_delta_log"
        log_files = []
        for path in sorted(log_dir.iterdir()):
            if path.is_file():
                log_files.append(InputFile(filename=path.name,
                                           data=path.read_bytes()))
        manifest_files = []
        for manifest_name in meta.get("manifests", ["existing_files.txt"]):
            manifest_path = sample_dir / manifest_name
            if not manifest_path.is_file():
                raise AnalyzerError(f"样例清单不存在：{manifest_path}")
            manifest_files.append(InputFile(
                filename=manifest_path.name,
                data=manifest_path.read_bytes(),
            ))
        report = _run_analysis(log_files, manifest_files,
                               from_version, to_version)
    except AnalyzerError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"读取样例失败：{exc}"}), 500
    return jsonify(report)


@app.errorhandler(413)
def too_large(_err):
    return render_template("index.html", samples=SAMPLES,
                           error="上传内容超过 256MB 上限"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
