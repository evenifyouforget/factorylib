"""Tests for factorylib.report (format_report/print_report), ported (and
split out) from factorylib.endfield.main's monolithic optimize()."""

from __future__ import annotations

from factorylib.fractions import snap_multiples
from factorylib.material import SOLID, Material, Recipe
from factorylib.optimize import solve
from factorylib.report import format_report, print_report


def _small_result():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(5 * ore, name="Supply", max_multiples=1)
    sell = Recipe(-1 * ore + 3 * dollar, name="Sell")
    result = solve(set(), [supply, sell], dollar)
    multiples = snap_multiples(result.multiples)
    return result, multiples


def test_format_report_contains_status_and_score():
    result, multiples = _small_result()
    text = format_report(result, multiples)
    assert "# Result 0:" in text
    assert "Maximized score: 15" in text


def test_format_report_lists_used_recipes():
    result, multiples = _small_result()
    text = format_report(result, multiples)
    assert "## Recipes Used" in text
    assert "multiples of Supply" in text
    assert "multiples of Sell" in text


def test_format_report_skips_zero_multiple_recipes():
    ore = Material(name="Ore", unit="/min", tags=SOLID)
    dollar = Material(name="$", unit="/min")
    supply = Recipe(6 * ore, name="Supply", max_multiples=1)
    worse = Recipe(-1 * ore + 1 * dollar, name="Worse Sell")
    better = Recipe(-1 * ore + 2 * dollar, name="Better Sell")
    result = solve(set(), [supply, worse, better], dollar)
    multiples = snap_multiples(result.multiples)
    text = format_report(result, multiples)
    assert "Worse Sell" not in text
    assert "Better Sell" in text


def test_format_report_balance_sheet_has_all_materials():
    result, multiples = _small_result()
    text = format_report(result, multiples)
    assert "## Balance Sheet Per Material" in text
    for material in result.materials:
        assert f"### {material.name}" in text


def test_print_report_prints_same_text(capsys):
    result, multiples = _small_result()
    print_report(result, multiples)
    captured = capsys.readouterr()
    assert captured.out.strip() == format_report(result, multiples).strip()
