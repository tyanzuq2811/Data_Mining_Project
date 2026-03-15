#!/usr/bin/env python
"""
run_papermill.py  –  Chạy tất cả notebooks theo thứ tự bằng papermill.

Usage
-----
    python scripts/run_papermill.py                # chạy tất cả
    python scripts/run_papermill.py --nb 01_eda    # chạy 1 notebook
"""

import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "outputs" / "executed_notebooks"


NOTEBOOKS = [
    "01_eda.ipynb",
    "02_preprocess_feature.ipynb",
    "03_mining_or_clustering.ipynb",
    "04_modeling.ipynb",
    "04b_semi_supervised.ipynb",
    "05_evaluation_report.ipynb",
]


def run_notebook(nb_name: str) -> None:
    """Execute one notebook via papermill."""
    try:
        import papermill as pm
    except ImportError:
        print("papermill chưa cài. Chạy:  pip install papermill")
        sys.exit(1)

    in_path = NB_DIR / nb_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / nb_name

    print(f">> Running {nb_name} ...")
    pm.execute_notebook(
        str(in_path),
        str(out_path),
        cwd=str(NB_DIR),
        kernel_name="python3",
    )
    print(f"  [OK] Saved -> {out_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Run notebooks with papermill.")
    parser.add_argument(
        "--nb",
        help="Chạy 1 notebook (ví dụ: 01_eda). Bỏ qua để chạy tất cả.",
    )
    args = parser.parse_args()

    if args.nb:
        # fuzzy match: user có thể gõ "01_eda" hoặc "01_eda.ipynb"
        target = args.nb if args.nb.endswith(".ipynb") else args.nb + ".ipynb"
        if target not in NOTEBOOKS:
            print(f"Notebook '{target}' không tìm thấy. Có: {NOTEBOOKS}")
            sys.exit(1)
        run_notebook(target)
    else:
        for nb in NOTEBOOKS:
            run_notebook(nb)

    print("\n[DONE] Completed.")


if __name__ == "__main__":
    main()
