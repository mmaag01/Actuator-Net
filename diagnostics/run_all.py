"""Run the four diagnostic scripts in sequence and concatenate their printed
summaries into `diagnostics/outputs/SUMMARY.txt`.

Scripts are executed as subprocesses so each one runs in exactly the same
environment it would if invoked directly. We scrape each child's stdout for
its summary block (delimited by the bars printed by `_common.save_summary`)
and append it to SUMMARY.txt in run order.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 for both the parent's console and the captured child output so
# Unicode characters (≥, ≈, ×, —) in the printed summaries work on Windows,
# where the default is cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "test1_torest_leakage.py",
    "test2_regime_residuals.py",
    "test3_accel_quality.py",
    "test4_model_agreement.py",
]

# Matches one boxed summary block printed by _common.save_summary.
BLOCK_RE = re.compile(r"=" * 72 + r"\n\[.*?\]\n" + r"=" * 72 + r"\n.*?(?=\n=+\n|\Z)",
                      re.DOTALL)


def extract_summary_block(stdout: str) -> str:
    matches = BLOCK_RE.findall(stdout)
    if not matches:
        return "(no summary block captured)\n"
    return matches[-1].rstrip() + "\n"


def main():
    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    all_blocks: list[str] = []
    header = (
        f"Actuator-Net diagnostics summary — generated {datetime.now().isoformat(timespec='seconds')}\n"
        f"Scripts: {', '.join(SCRIPTS)}\n"
    )
    all_blocks.append(header)

    for script in SCRIPTS:
        print(f"\n>>> Running {script} …")
        proc = subprocess.run(
            [sys.executable, str(HERE / script)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            block = (f"{'=' * 72}\n[{script}] FAILED (exit {proc.returncode})\n"
                     f"{'=' * 72}\n{proc.stderr.strip()}\n")
        else:
            block = extract_summary_block(proc.stdout)
        all_blocks.append(block)

    summary_path.write_text("\n".join(all_blocks), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
