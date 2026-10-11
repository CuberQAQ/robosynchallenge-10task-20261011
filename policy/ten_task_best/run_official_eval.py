"""Run the unchanged official evaluator and avoid native shutdown GC on ACT.

Exceptions and signals during evaluation propagate normally. Hard termination
is reached only after the evaluator's main function has returned successfully,
including its metrics file write and result summary.
"""
import os
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    evaluator = Path(__file__).resolve().parents[2] / "scripts" / "eval_policy.py"
    sys.argv[0] = str(evaluator)
    official_globals = runpy.run_path(str(evaluator), run_name="__main__")
    print("OFFICIAL_EVALUATOR_RETURNED: ending ACT process before native shutdown GC", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
