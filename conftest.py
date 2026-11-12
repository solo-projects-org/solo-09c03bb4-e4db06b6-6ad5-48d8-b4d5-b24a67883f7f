"""pytest 引导：隔离模式（python -I）下补回仓库根目录，并确保样例存在。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _ensure_samples() -> None:
    sample_dirs = ("healthy_table", "incident_table",
                   "encoding_table", "legacy_manifest_table")
    missing = [name for name in sample_dirs
               if not any((ROOT / "samples" / name / "_delta_log").glob("*.json"))]
    if not missing:
        return
    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    # 隔离模式下没有 site-packages 之外的路径问题，但生成脚本依赖 pyarrow；
    # 用户可能只想跑部分测试，因此样例缺失时必须成功生成。
    import importlib
    generate_samples = importlib.import_module("generate_samples")
    generate_samples.main()


_ensure_samples()
