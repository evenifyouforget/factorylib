from unittest.mock import patch

from factorylib.endfield import diagram
from factorylib.endfield.diagram import _build_graph, generate_diagram
from factorylib.endfield.wuling import WulingConfig, build_formulas

_RESOURCE_NAMES = ["a", "b", "c"]
_RESOURCE_LABELS = {"a": "Resource A", "b": "Resource B", "c": "Resource C"}
_FORMULA_LABELS = {"make_b": "Make B from A"}


def _formula(consumption):
    from factorylib.optimize import Formula

    return Formula(consumption=consumption, output=0)


def test_build_graph_includes_active_formula_and_touched_resources():
    import numpy as np

    formulas = {"make_b": _formula(np.array([30.0, -30.0, 0.0]))}
    dot = _build_graph(
        {"make_b": 2.0}, formulas, _RESOURCE_NAMES, _RESOURCE_LABELS, _FORMULA_LABELS
    )
    source = dot.source
    assert "Resource A" in source
    assert "Resource B" in source
    assert "Resource C" not in source  # untouched by any active formula
    assert "Make B from A" in source
    assert "60/min" in source  # 2 multiples * 30/multiple


def test_build_graph_excludes_zero_rate_formulas():
    import numpy as np

    formulas = {"make_b": _formula(np.array([30.0, -30.0, 0.0]))}
    dot = _build_graph(
        {"make_b": 0.0}, formulas, _RESOURCE_NAMES, _RESOURCE_LABELS, _FORMULA_LABELS
    )
    assert "Make B from A" not in dot.source
    assert "Resource A" not in dot.source


def test_generate_diagram_returns_none_when_graphviz_unavailable():
    with patch.object(diagram, "graphviz", None):
        result = generate_diagram({}, {}, _RESOURCE_NAMES, {}, {}, "/tmp/out.png")
    assert result is None


def test_generate_diagram_returns_none_for_empty_plan():
    """Must short-circuit before ever calling render() -- regression for
    a real bug where the "any active formulas?" check looked at
    dot.body, which is never empty (dot.attr(rankdir=...) always adds a
    line), so this case fell through to render() and only accidentally
    returned None via the ExecutableNotFound path."""
    with patch.object(diagram.graphviz.Digraph, "render") as mock_render:
        result = generate_diagram(
            {"idle": 0.0},
            {},
            _RESOURCE_NAMES,
            _RESOURCE_LABELS,
            _FORMULA_LABELS,
            "/tmp/out.png",
        )
    assert result is None
    mock_render.assert_not_called()


def test_generate_diagram_returns_none_when_dot_executable_missing():
    """Regression for the "installed the Python package but not the
    actual Graphviz binaries" case -- must degrade gracefully, not
    raise."""
    import numpy as np

    formulas = {"make_b": _formula(np.array([30.0, -30.0, 0.0]))}
    not_found = diagram.graphviz.ExecutableNotFound("dot")
    with patch.object(diagram.graphviz.Digraph, "render", side_effect=not_found):
        result = generate_diagram(
            {"make_b": 1.0},
            formulas,
            _RESOURCE_NAMES,
            _RESOURCE_LABELS,
            _FORMULA_LABELS,
            "/tmp/out.png",
        )
    assert result is None


def test_generate_diagram_real_wuling_formulas_does_not_raise():
    """End-to-end smoke test against the real formula set -- exercises
    the full path (build_formulas -> _build_graph) without requiring a
    working `dot` executable (whatever this environment has/lacks is
    fine; the point is no exception escapes)."""
    from factorylib.endfield.wuling import (
        FORMULA_LABELS,
        RESOURCE_LABELS,
        RESOURCE_NAMES,
    )

    formulas = build_formulas(WulingConfig())
    rates_by_name = {"cup_conv": 8.0, "sc": 2.0, "ya": 0.0}
    result = generate_diagram(
        rates_by_name,
        formulas,
        RESOURCE_NAMES,
        RESOURCE_LABELS,
        FORMULA_LABELS,
        "/tmp/factorylib_test_diagram.png",
    )
    assert result is None or isinstance(result, str)
