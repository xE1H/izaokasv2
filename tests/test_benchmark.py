"""That the benchmark is wired to the leaderboard, checked without Isaac Sim.

``lituanicax_sdk/benchmark.py`` launches the simulator before it can be
imported, so these read its source instead: the module compiles, the flags a
team types exist, and the publishing step is genuinely reached from ``main()``
rather than merely defined. The behaviour of what it calls is covered by
``test_submit.py``, which exercises the real client against a real socket.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parent.parent / "lituanicax_sdk" / "benchmark.py"


@pytest.fixture(scope="module")
def module() -> ast.Module:
    return ast.parse(BENCHMARK.read_text())


def function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"benchmark.py defines no {name}()")


def calls_in(node: ast.AST) -> set[str]:
    """Every plain function call made anywhere inside ``node``."""
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def cli_flags(module: ast.Module) -> set[str]:
    """The long options passed to ``parser.add_argument``."""
    flags = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags.update(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
    return flags


def test_the_benchmark_compiles(tmp_path):
    py_compile.compile(str(BENCHMARK), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_a_scored_run_publishes(module):
    assert "publish" in calls_in(function(module, "main"))


def test_publishing_uses_the_sdk_client(module):
    assert calls_in(function(module, "publish")) >= {"submit", "print_outcome"}


def test_the_report_is_written_before_it_is_published(module):
    """A board that is down must not cost a team its submission.json."""
    body = calls_in(function(module, "main"))
    assert {"write_report", "publish"} <= body

    order = [
        node.value.func.id
        for node in function(module, "main").body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert order.index("write_report") < order.index("publish")


@pytest.mark.parametrize("flag", ["--no-submit", "--team"])
def test_the_publishing_flags_exist(module, flag):
    assert flag in cli_flags(module)


def test_no_submit_skips_publishing(module):
    """The flag has to be read, not merely accepted."""
    publish = function(module, "publish")
    guarded = [
        node
        for node in publish.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "no_submit"
            for child in ast.walk(node.test)
        )
    ]
    assert guarded, "publish() ignores --no-submit"
    assert any(isinstance(child, ast.Return) for child in ast.walk(guarded[0]))


def test_shutdown_cannot_hold_the_process_open(module):
    """Isaac Sim's teardown does not reliably return; the run must end anyway."""
    assert "close_or_bail" in calls_in(function(module, "main"))

    bail = function(module, "close_or_bail")
    source = ast.dump(bail)
    assert "Thread" in source, "close_or_bail() arms no watchdog"
    assert "_exit" in source, "a stuck teardown needs os._exit, not sys.exit"


def test_the_result_is_safe_before_the_deadline_is_armed(module):
    """Nothing may be lost when the watchdog fires, so it is armed last."""
    order = [
        node.value.func.id
        for node in function(module, "main").body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert order.index("write_report") < order.index("close_or_bail")
    assert order.index("publish") < order.index("close_or_bail")
