"""
Tests for the pipeline entry point: step order, resuming, argument passing,
and that every module accepts the arguments the pipeline sends it.
"""

import pytest

from energy_dispatch import dashboard, data, evaluate, pipeline, planning, sensitivity


@pytest.fixture
def calls(monkeypatch):
    """Replace every step's main with a recorder."""
    recorded = []
    fakes = {name: (lambda argv, name=name: recorded.append((name, argv)))
             for name in pipeline.STEP_NAMES}  # fmt: skip
    monkeypatch.setattr(pipeline, "_step_mains", lambda: fakes)
    return recorded


def test_default_run_executes_every_stage_in_order(calls):
    pipeline.main([])
    assert [name for name, _ in calls] == [
        "data", "forecast", "evaluate", "sensitivity", "planning", "diagnostics",
    ]  # fmt: skip
    assert all(argv == [] for _, argv in calls)


def test_resume_and_export(calls):
    pipeline.main(["--from-step", "planning", "--export-demo-results"])
    assert calls == [
        ("planning", []), ("diagnostics", []), ("export", ["--export-demo-results"]),
    ]  # fmt: skip


def test_only_and_argument_propagation(calls):
    pipeline.main(["--only", "data", "evaluate", "--skip-download", "--max-days", "7"])
    assert calls == [("data", ["--skip-download"]), ("evaluate", ["--max-days", "7"])]


def test_every_module_accepts_the_arguments_the_pipeline_sends(monkeypatch):
    args = pipeline.build_arg_parser().parse_args(
        ["--skip-download", "--max-days", "3", "--export-demo-results"]
    )
    parsers = {
        "data": data._build_arg_parser(),
        "evaluate": evaluate._build_arg_parser(),
        "sensitivity": sensitivity._build_arg_parser(),
        "planning": planning._build_arg_parser(),
    }
    for step, parser in parsers.items():
        parser.parse_args(pipeline.step_arguments(step, args))  # would exit on a bad flag

    exported = []
    monkeypatch.setattr(dashboard, "export_demo_results", lambda: exported.append(True) or [])
    dashboard.main(pipeline.step_arguments("export", args))
    assert exported == [True]


def test_diagnostics_accepts_max_days(monkeypatch):
    from energy_dispatch import diagnostics

    seen = {}

    def stop(path, *a, **k):  # stop right after argument parsing
        seen["parsed"] = True
        raise SystemExit(0)

    monkeypatch.setattr(diagnostics.pd, "read_parquet", stop)
    with pytest.raises(SystemExit):
        diagnostics.main(["--max-days", "3"])
    assert seen == {"parsed": True}
