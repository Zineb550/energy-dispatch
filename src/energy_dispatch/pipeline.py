"""
Single entry point: regenerates every result end to end (spec section 12).

    python -m energy_dispatch.pipeline                      # everything
    python -m energy_dispatch.pipeline --skip-download      # reuse data/raw/
    python -m energy_dispatch.pipeline --from-step evaluate # resume later
    python -m energy_dispatch.pipeline --max-days 7         # quick smoke run

Steps, in order (each is the module's own command, run with the same code
path as `python -m energy_dispatch.<module>`):

    data         download + clean OPSD data   -> spain_hourly.parquet
    forecast     rolling-origin backtest      -> forecast table + metrics
    evaluate     8 forecast x battery plans   -> scenario results
    sensitivity  battery value vs fleet       -> sensitivity results
    planning     quantile plans (tau*)        -> planning results
    diagnostics  DM tests, SHAP, error slices -> diagnostics results
    export       (optional, --export-demo-results) copy results for the hosted app

Nothing is re-implemented here; the pipeline only sequences the stages.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

from energy_dispatch import (
    dashboard,
    data,
    diagnostics,
    evaluate,
    forecast,
    planning,
    sensitivity,
)

logger = logging.getLogger(__name__)

STEP_NAMES: tuple[str, ...] = (
    "data", "forecast", "evaluate", "sensitivity", "planning", "diagnostics", "export",
)  # fmt: skip


def _step_mains() -> dict[str, Callable[[list[str]], None]]:
    # Looked up at call time so tests can substitute a module's main.
    return {
        "data": data.main,
        "forecast": forecast.main,
        "evaluate": evaluate.main,
        "sensitivity": sensitivity.main,
        "planning": planning.main,
        "diagnostics": diagnostics.main,
        "export": dashboard.main,
    }


def step_arguments(step: str, args: argparse.Namespace) -> list[str]:
    """Command-line arguments passed to one step's main()."""
    argv: list[str] = []
    if step == "data" and args.skip_download:
        argv.append("--skip-download")
    if step in ("evaluate", "sensitivity", "planning", "diagnostics") and args.max_days:
        argv += ["--max-days", str(args.max_days)]
    if step == "export":
        argv.append("--export-demo-results")
    return argv


def planned_steps(args: argparse.Namespace) -> list[str]:
    steps = [s for s in STEP_NAMES if s != "export" or args.export_demo_results]
    if args.from_step:
        steps = steps[steps.index(args.from_step) :]
    if args.only:
        steps = [s for s in steps if s in args.only]
    return steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the whole project end to end.")
    parser.add_argument(
        "--from-step", choices=[s for s in STEP_NAMES if s != "export"],
        help="Start at this step (earlier results are reused from data/processed/).",
    )  # fmt: skip
    parser.add_argument(
        "--only", nargs="+", choices=STEP_NAMES, help="Run only these steps, in pipeline order."
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Use the CSVs already in data/raw/."
    )
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Limit evaluation, sensitivity, planning and diagnostics to N days (smoke run).",
    )  # fmt: skip
    parser.add_argument(
        "--export-demo-results", action="store_true",
        help="Finish by copying results to app/demo_results/ for the hosted dashboard.",
    )  # fmt: skip
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    steps = planned_steps(args)
    mains = _step_mains()
    timings = []
    for i, step in enumerate(steps, start=1):
        step_argv = step_arguments(step, args)
        logger.info("\n=== Step %d/%d: %s %s ===", i, len(steps), step, " ".join(step_argv))
        start = time.perf_counter()
        mains[step](step_argv)
        timings.append((step, time.perf_counter() - start))
    logger.info("\nPipeline finished:")
    for step, seconds in timings:
        logger.info("  %-12s %6.1f min", step, seconds / 60)


if __name__ == "__main__":
    main()
