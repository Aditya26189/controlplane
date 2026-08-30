"""Every call a script makes into ``controlplane/`` must bind to a real signature.

Written after a 160-minute Kaggle run failed on its final stage because
``scripts/13_pilot_run.py`` called ``load_model(config)`` where the signature is
``load_model(name: str, *, quantization=...)``. The extraction, every validation
stage and the warrant matrix had all completed; the pilot then died at its first
GPU call and took 2.7 GPU-hours with it.

**The exemption was the mistake, not the typo.** ``13_pilot_run.py`` is exempt
from ``test_every_script_has_a_smoke_test`` because it needs a GPU -- which is
true of what it *computes*, and false of how it is *wired*. Binding an argument
list to a signature needs no GPU, no model and no network, and would have caught
this in milliseconds.

So this checks statically, across every script: walk the AST, find calls to
things imported from ``controlplane``, and bind the call to the real signature.
It catches wrong argument counts, unknown keywords and missing required
arguments in scripts nothing else can execute cheaply.

What it deliberately does NOT check is types -- ``load_model(config)`` binds
positionally against ``name: str`` and would still pass here on arity alone. So
a companion check reads the annotations: a call passing a ``Config`` object
where the parameter is annotated ``str`` is the specific error that happened,
and it is worth naming rather than generalising away.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    p for p in (PROJECT_ROOT / "scripts").glob("*.py") if not p.name.startswith("_")
)


def _controlplane_imports(tree: ast.Module) -> dict[str, str]:
    """Map local name -> dotted target, for names imported from controlplane."""
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or not node.module.startswith("controlplane"):
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                imported[local] = f"{node.module}.{alias.name}"
    return imported


def _resolve(dotted: str):
    """Import the module and return the attribute, or None if not a function."""
    module_path, _, name = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception:  # pragma: no cover - an unimportable module is another test
        return None
    target = getattr(module, name, None)
    if target is None or not callable(target):
        return None
    if inspect.isclass(target):
        return None
    return target


def _calls(tree: ast.Module):
    """Every direct call by bare name, with its AST node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node.func.id, node


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_controlplane_call_binds_to_its_signature(script: Path) -> None:
    """Arity and keyword names, checked without running anything.

    A call with the wrong number of arguments, or a keyword the function does
    not accept, fails here instead of an hour into a GPU session.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported = _controlplane_imports(tree)
    problems: list[str] = []

    for name, node in _calls(tree):
        dotted = imported.get(name)
        if dotted is None:
            continue
        target = _resolve(dotted)
        if target is None:
            continue
        # *args / **kwargs at the call site make binding unknowable statically.
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue
        args = [object()] * len(node.args)
        kwargs = {k.arg: object() for k in node.keywords}
        try:
            inspect.signature(target).bind(*args, **kwargs)
        except TypeError as exc:
            problems.append(
                f"  line {node.lineno}: {name}(...) does not bind to "
                f"{dotted}{inspect.signature(target)} -- {exc}"
            )

    assert not problems, (
        f"{script.name} calls controlplane with arguments that do not bind:\n"
        + "\n".join(problems)
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_script_passes_the_config_object_where_a_name_is_expected(
    script: Path,
) -> None:
    """The exact error that cost 2.7 GPU-hours.

    ``load_model(config)`` binds fine on arity -- ``config`` lands in
    ``name: str`` -- and then transformers tries to resolve a repo id from the
    repr of a dataclass. Arity checking alone would not have caught it, so the
    annotation is checked for the one variable name every script uses for the
    resolved config.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported = _controlplane_imports(tree)
    problems: list[str] = []

    for name, node in _calls(tree):
        dotted = imported.get(name)
        if dotted is None:
            continue
        target = _resolve(dotted)
        if target is None:
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        parameters = list(signature.parameters.values())

        for index, arg in enumerate(node.args):
            if not (isinstance(arg, ast.Name) and arg.id == "config"):
                continue
            if index >= len(parameters):
                continue
            annotation = parameters[index].annotation
            if annotation is inspect.Parameter.empty:
                continue
            rendered = getattr(annotation, "__name__", str(annotation))
            if "Config" in rendered:
                continue
            problems.append(
                f"  line {node.lineno}: {name}(config, ...) passes the resolved "
                f"Config into parameter {parameters[index].name!r}, annotated "
                f"{rendered!r}. Pass the field, not the config object."
            )

    assert not problems, (
        f"{script.name} passes the config object where a value is expected:\n"
        + "\n".join(problems)
    )


def test_the_guard_catches_the_bug_it_was_written_for(tmp_path: Path) -> None:
    """The regression, reconstructed. Without this the guard could rot silently."""
    bad = tmp_path / "bad_script.py"
    bad.write_text(
        "from controlplane.extract.model import load_model\n"
        "def main(config):\n"
        "    return load_model(config)\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="passes the resolved Config"):
        test_no_script_passes_the_config_object_where_a_name_is_expected(bad)


def test_the_guard_catches_a_wrong_keyword(tmp_path: Path) -> None:
    bad = tmp_path / "bad_kwarg.py"
    bad.write_text(
        "from controlplane.extract.model import load_model\n"
        "def main():\n"
        '    return load_model("x", quantisation="nf4")\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="does not bind"):
        test_every_controlplane_call_binds_to_its_signature(bad)


def test_the_guard_actually_inspects_something() -> None:
    """A guard that resolved no calls would pass vacuously on every script."""
    tree = ast.parse(
        (PROJECT_ROOT / "scripts" / "13_pilot_run.py").read_text(encoding="utf-8")
    )
    imported = _controlplane_imports(tree)
    resolved = [n for n, _ in _calls(tree) if _resolve(imported.get(n, "")) is not None]
    assert len(resolved) >= 5, (
        f"only {len(resolved)} controlplane calls resolved in 13_pilot_run.py; "
        "the guard is not seeing what it claims to check"
    )
