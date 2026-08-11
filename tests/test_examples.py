"""Tier three: every example script runs to completion under a reduced workload.

The scripts are loaded from their path and their ``main`` is called directly,
rather than launched as subprocesses. That keeps the tier fast and, more useful
in practice, surfaces a failure as the real traceback instead of a non-zero exit
code with the message buried in captured output.

Each script accepts ``--quick``, which shortens the scenario and cuts the Monte
Carlo run count. The point of this tier is that the wiring holds together, not
that the numbers are good; the numbers are tiers one and two.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(path.name for path in EXAMPLES.glob("*.py"))

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
# The README embeds these, and a repository that ships a heavy figure directory
# is a repository people clone slowly. Kept in step with the budget the
# portfolio validator applies.
FIGURE_BUDGET_BYTES = 250 * 1024


def _load(name: str) -> ModuleType:
    """Import an example script by path under a private module name."""
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def outputs(tmp_path: Path) -> Iterator[Path]:
    """Give each script a private output directory so tests do not collide."""
    target = tmp_path / "outputs"
    target.mkdir()
    yield target


def test_the_expected_scripts_are_present() -> None:
    """Guard the guard: the tier is worthless if it discovers nothing."""
    assert SCRIPTS == [
        "asynchronous_fusion.py",
        "compare_filters.py",
        "consistency_study.py",
        "ekf_versus_ukf.py",
        "make_figures.py",
    ]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_runs_to_completion(
    name: str, outputs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each script returns zero and prints something under a reduced workload."""
    module = _load(name)
    arguments = ["--quick"]
    if "--outdir" in Path(EXAMPLES / name).read_text(encoding="utf-8"):
        arguments += ["--outdir", str(outputs)]
    assert module.main(arguments) == 0
    captured = capsys.readouterr().out
    assert captured.strip(), "a script that prints nothing has told the reader nothing"


@pytest.mark.parametrize("name", ["compare_filters.py", "consistency_study.py", "make_figures.py"])
def test_figure_writing_scripts_produce_files(name: str, outputs: Path) -> None:
    """Scripts that claim to write figures must actually write them."""
    module = _load(name)
    assert module.main(["--quick", "--outdir", str(outputs)]) == 0
    written = sorted(path.name for path in outputs.glob("*.png"))
    assert written, "no figure was written"
    for path in outputs.glob("*.png"):
        assert path.stat().st_size > 1000, f"{path.name} is implausibly small"


def test_the_tracked_figures_exist_and_fit_their_budget() -> None:
    """The figures the README embeds are committed, and small enough to stay committed.

    They are snapshots, not build artefacts. Matplotlib output is not byte
    reproducible across platforms or across its own releases, so nothing here
    compares them byte for byte against a fresh run; that check would fail for
    reasons unrelated to this package. What is checked is that they are present,
    that they are plausible files, and that regenerating them has not quietly
    pushed the repository over its figure budget.
    """
    figures = sorted((FIGURES).glob("*.png"))
    assert [path.name for path in figures] == [
        "nees-ekf-ukf.png",
        "nees-specification.png",
        "tracks.png",
    ]
    total = sum(path.stat().st_size for path in figures)
    assert all(path.stat().st_size > 5_000 for path in figures)
    assert total < FIGURE_BUDGET_BYTES, f"tracked figures total {total} bytes"
