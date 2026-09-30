"""Run the complete Ix-construction comparison matrix.

Place/run this file from FANQO/user on branch Ixcononly:

    python compare_all.py

Conventions
-----------
For every call

    fq.compare(name1, name2)

the plot uses:

    RED  = name1 has smaller tracked Ix drift
    BLUE = name2 has smaller tracked Ix drift

Full 5-D comparisons use the configured (x0,y0) tracking grid.
Any comparison containing a_box_y0 automatically runs only on y0=0.
"""

from __future__ import annotations

from pathlib import Path
import csv
import sys

# Always prefer the FANQO source from this local checkout:
#
# FANQO/
#   src/fanqo/
#   user/compare_all.py
#
# This prevents an older pip-installed FANQO from being imported by mistake.
_THIS_FILE = Path(__file__).resolve()
_LOCAL_SRC = _THIS_FILE.parent.parent / "src"
if (_LOCAL_SRC / "fanqo").is_dir():
    sys.path.insert(0, str(_LOCAL_SRC))

import fanqo as fq


CONFIG_FILE = "general_config.py"

# Keep the ordering explicit because it also defines the plot color convention.
FULL_5D_COMPARISONS = [
    ("a_box", "hybrid"),
    ("a_box", "eigen"),
    ("hybrid", "eigen"),
]

Y0_COMPARISONS = [
    ("a_box_y0", "a_box"),
    ("a_box_y0", "hybrid"),
    ("a_box_y0", "eigen"),
]


def _summary_row(name1, name2, result):
    stats = result["stats"]
    return {
        "name1_red": name1,
        "name2_blue": name2,
        "horizontal_slice_only": stats["horizontal_slice_only"],
        "physical_ring_cells": stats["physical_ring_cells"],
        "turns": stats["turns"],
        "valid_both": stats["valid_both"],
        "name1_better": stats["method1_better"],
        "name2_better": stats["method2_better"],
        "ties": stats["ties"],
        "fraction_name1_better": stats["fraction_method1_better"],
        "fraction_name2_better": stats["fraction_method2_better"],
        "median_score": stats["median_score"],
        "plot": str(result["plot_path"]),
        "data_csv": str(result["csv_path"]),
    }


def main():
    # Load one fixed lattice. Every constructor below uses this same machine.
    fq.load(CONFIG_FILE, force=True)

    print("Using FANQO from:")
    print(Path(fq.__file__).resolve())
    print()

    print("=" * 88)
    print("IX CONSTRUCTION COMPARISON")
    print("=" * 88)
    print(fq.status())
    print()

    # Construct everything once up front. compare() will reuse the cached Ix.
    print("Constructing invariants...")
    for method in fq.available_methods():
        result = fq.construct(method)
        print(f"  {method:<10} : {result['details']}")

    comparisons = FULL_5D_COMPARISONS + Y0_COMPARISONS
    summary = []

    for index, (name1, name2) in enumerate(comparisons, start=1):
        print()
        print("=" * 88)
        print(f"[{index}/{len(comparisons)}] {name1} vs {name2}")
        print(f"RED = {name1} better | BLUE = {name2} better")
        if "a_box_y0" in (name1, name2):
            print("Tracking domain: y0 = 0 only")
        else:
            print("Tracking domain: full configured (x0, y0) grid")
        print("=" * 88)

        result = fq.compare(name1, name2)
        row = _summary_row(name1, name2, result)
        summary.append(row)

        print(f"valid points       : {row['valid_both']}")
        print(f"{name1} better      : {row['name1_better']}"
              f" ({row['fraction_name1_better']:.3f})")
        print(f"{name2} better      : {row['name2_better']}"
              f" ({row['fraction_name2_better']:.3f})")
        print(f"ties               : {row['ties']}")
        print(f"median score       : {row['median_score']}")
        print(f"plot               : {row['plot']}")
        print(f"pointwise data     : {row['data_csv']}")

    # Every comparison folder lives directly under the configured output root.
    # Derive that root from the last public result instead of using private API.
    output_root = Path(result["output_directory"]).parent
    summary_path = output_root / "all_method_comparisons.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 88)
    print("ALL COMPARISONS FINISHED")
    print("=" * 88)
    print("Summary:", summary_path)
    print()
    print("Interpretation:")
    print("  score = log10(D_name2) - log10(D_name1)")
    print("  positive -> name1 better -> red")
    print("  negative -> name2 better -> blue")
    print("=" * 88)


if __name__ == "__main__":
    main()
