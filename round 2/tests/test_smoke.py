"""Every script actually runs. ``SPEC.md`` §10, ``test_smoke``.

**Why this exists, written after the second time it was needed.** The test suite
imports from ``src/`` and never executes a script, so a script can be broken for
a whole phase without a single test failing. It has now happened twice:

* ``scripts/02_validate.py`` referenced ``RecordKind`` without importing it, for
  three commits. The guard was added to two scripts and the import to one.
* the same shape produced the ``fpr_hard_negatives`` conflation
  (``DECISIONS.md`` 040): a fix applied at one call site when the cause was a
  shared concept.

Scripts are where ``src/`` is wired together, and wiring is exactly what unit
tests do not exercise. These run each one end to end at the smallest size that
still does real work, and assert the artifacts appear.

They are slow by the standards of the rest of the suite — tens of seconds each —
and that is the price of the only test that would have caught either bug.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.smoke


def run_script(name: str, *args: str, out: Path) -> subprocess.CompletedProcess:
    """Run a script in a subprocess with its own results directory.

    A subprocess rather than an import, because the failure mode is a missing
    import at module scope and importing the module here would mask exactly
    that. The script has to be executed the way a person executes it.
    """
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / name),
        "--config",
        str(PROJECT_ROOT / "config.yaml"),
        "--out",
        str(out),
        *args,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=900,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        raise AssertionError(
            f"scripts/{name} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )
    return result


def test_smoke_build_evalsets(tmp_path: Path) -> None:
    """The eval sets build, freeze, register and verify."""
    run_script(
        "01_build_evalsets.py", "--evalsets-out", str(tmp_path / "evalsets"),
        out=tmp_path,
    )
    payload = json.loads((tmp_path / "evalset_validation.json").read_text(encoding="utf-8"))
    assert payload["runs"], "no eval set was scored"
    for run in payload["runs"]:
        assert run["warrant_status"] in ("VALID", "REFUSED")
        assert run["n_items"] > 0


def test_smoke_validate_fixture(tmp_path: Path) -> None:
    """The tier ablation runs and writes a ladder with intervals."""
    run_script("02_validate.py", "--fixture", "--smoke", out=tmp_path)
    ladder = json.loads((tmp_path / "tier_ladder-fixture.json").read_text(encoding="utf-8"))
    assert ladder["rungs"], "no rungs measured"
    for rung in ladder["rungs"].values():
        assert len(rung["controls"]) == 5
        auroc = rung["metrics"].get("auroc")
        if auroc is not None:
            assert auroc["ci_low"] <= auroc["value"] <= auroc["ci_high"]
    assert (tmp_path / "tier_ladder-fixture.png").exists()


def test_smoke_matrix(tmp_path: Path) -> None:
    """The matrix populates, routes, and renders RESULTS.md with its refusal."""
    run_script("03_matrix.py", "--smoke", out=tmp_path)

    matrix = json.loads((tmp_path / "warrant_matrix.json").read_text(encoding="utf-8"))
    summary = matrix["matrix"]["summary"]
    assert sum(summary.values()) == len(matrix["matrix"]["detectors"]) * len(
        matrix["matrix"]["envelopes"]
    ), "not every cell is accounted for"
    assert summary["UNVALIDATED"] > 0, "no cell is unvalidated; the axes are wrong"
    assert matrix["routing"], "no routing decisions recorded"

    results = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "# RESULTS" in results
    # The fixture refusal must be present, since a smoke run is all fixtures.
    assert "FIXTURE" in results
    assert "Outstanding measurement" in results


def test_every_script_has_a_smoke_test() -> None:
    """A new script without a smoke test is the gap this file exists to close.

    Excludes the notebook builder, which is covered by
    ``test_notebook_is_generated_from_its_script``.
    """
    scripts = {
        path.name
        for path in (PROJECT_ROOT / "scripts").glob("*.py")
        if not path.name.startswith("_")
    }
    covered = {"01_build_evalsets.py", "02_validate.py", "03_matrix.py"}
    exempt = {
        # Needs a GPU; its wiring is checked by the notebook's own self-check
        # and by tests/test_extraction.py for everything that runs on CPU.
        "00_extract.py",
        "build_notebooks.py",
    }
    uncovered = scripts - covered - exempt
    assert not uncovered, (
        f"scripts without a smoke test: {sorted(uncovered)}. Scripts are where "
        "src/ is wired together, and wiring is what unit tests do not exercise."
    )


def test_notebook_is_generated_from_its_script(tmp_path: Path) -> None:
    """The committed notebook matches what its generator produces.

    Notebook JSON is not reviewable in a diff, so the generator is the source of
    truth. A hand-edited notebook would drift from it silently, and the drift
    would only surface on a GPU session an hour into a run.
    """
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_notebooks.py"),
         "--out", str(tmp_path)],
        check=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
    )
    generated = json.loads((tmp_path / "run_on_kaggle.ipynb").read_text(encoding="utf-8"))
    committed = json.loads(
        (PROJECT_ROOT / "notebooks" / "run_on_kaggle.ipynb").read_text(encoding="utf-8")
    )
    assert [c["source"] for c in generated["cells"]] == [
        c["source"] for c in committed["cells"]
    ], (
        "notebooks/run_on_kaggle.ipynb differs from what build_notebooks.py "
        "produces. Edit the script and regenerate; never hand-edit the notebook."
    )


def test_every_notebook_code_cell_compiles() -> None:
    """Every code cell in the generated notebook is valid Python.

    Written after shipping a notebook whose repo-detection cell was a syntax
    error. Two bugs stacked, and neither is visible in a notebook diff:

    * ``source`` entries carried no trailing newline. nbformat defines the field
      as a list that **concatenates** to the cell body, so a reader that
      concatenates runs the whole cell onto one line, while Kaggle — which joins
      with newlines — does not. The same notebook is therefore fine in one place
      and a syntax error in another.
    * An escaped newline inside an error message became a *real* newline in the
      generated source, splitting string literals across lines. That one is a
      syntax error everywhere, and it survived a review of the generator.

    Compiling the output is the only check that would have caught either.

    IPython magics are not Python; they are replaced with ``pass`` **preserving
    indentation**, because a magic inside an ``if`` block would otherwise break
    the block rather than the cell — which is a bug in the checker that looks
    like a bug in the notebook.
    """
    import ast
    import re

    notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "run_on_kaggle.ipynb").read_text(encoding="utf-8")
    )

    def neutralise(line: str) -> str:
        match = re.match(r"^(\s*)[!%]", line)
        return match.group(1) + "pass" if match else line

    failures = []
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells, "the notebook has no code cells"

    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        # Try the source as written FIRST. Neutralising magics unconditionally
        # produced a false positive: a format continuation like
        #     "unsupported GPU: %s, sm %d.%d"
        #     % (name, major, minor),
        # begins a line with "%" and was rewritten to "pass", breaking an
        # expression that had been valid all along. IPython treats "%" as a
        # magic only at the start of a statement, never as a bracketed
        # continuation. A cell that parses as written needs no rewriting; only
        # one that fails can hold a magic, and there the rewrite is worth trying.
        try:
            ast.parse(source)
            continue
        except SyntaxError:
            pass
        cleaned = "\n".join(neutralise(line) for line in source.split("\n"))
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            failures.append("code cell %d: %s at line %s" % (index, exc.msg, exc.lineno))

    assert not failures, "notebook cells do not compile:\n  " + "\n  ".join(failures)


def test_notebook_source_entries_keep_their_newlines() -> None:
    """nbformat requires ``source`` entries to concatenate to the cell body.

    Checked separately from compilation, because a malformed notebook can still
    compile under a reader that joins with newlines — and the malformed version
    is the one that breaks somewhere else.
    """
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "run_on_kaggle.ipynb").read_text(encoding="utf-8")
    )
    for index, cell in enumerate(notebook["cells"]):
        source = cell["source"]
        if len(source) < 2:
            continue
        missing = [i for i, line in enumerate(source[:-1]) if not line.endswith("\n")]
        assert not missing, (
            "cell %d: %d source entries lack a trailing newline. A reader that "
            "concatenates rather than joins will run the cell onto one line."
            % (index, len(missing))
        )


def test_notebook_has_no_stray_real_newlines_in_string_literals() -> None:
    """The specific corruption: an escape that became a literal line break.

    A string literal opened on one line and closed on the next is the signature.
    Compilation already catches it, but this names the cause in the failure
    message rather than reporting "invalid syntax" and leaving the reader to
    find it.
    """
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "run_on_kaggle.ipynb").read_text(encoding="utf-8")
    )
    offenders = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for line_number, line in enumerate(cell["source"], start=1):
            stripped = line.rstrip("\n")
            # An odd number of unescaped double quotes means the literal is
            # still open when the line ends.
            unescaped = stripped.replace('\\"', "")
            if unescaped.count('"') % 2 == 1 and '"""' not in stripped:
                offenders.append("cell %d line %d: %r" % (index, line_number, stripped))
    assert not offenders, (
        "string literals left open at end of line -- an escaped newline probably "
        "became a real one in the generator:\n  " + "\n  ".join(offenders)
    )


def test_extraction_script_imports_cleanly() -> None:
    """The GPU script's module-scope imports resolve without a GPU.

    Cannot run it end to end here, but the bug this file was written for was a
    NameError at module scope, and that is checkable without a card.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "import importlib.util as u; "
         "spec = u.spec_from_file_location('extract_cli', 'scripts/00_extract.py'); "
         "m = u.module_from_spec(spec); spec.loader.exec_module(m); "
         "print(m.parse_args(['--smoke']).smoke)"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=180,
    )
    assert result.returncode == 0, (
        f"scripts/00_extract.py does not import cleanly:\n{result.stderr[-2000:]}"
    )
    assert "True" in result.stdout
