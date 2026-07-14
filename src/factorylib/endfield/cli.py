"""CLI: compute an optimal Wuling production plan and print it, along with
any tied alternatives (see factorylib.alternatives)."""

from __future__ import annotations

import argparse

import numpy as np

from factorylib.alternatives import find_alternatives
from factorylib.endfield.wuling import (
    RESOURCE_NAMES,
    XI_PER_FORGE,
    WulingConfig,
    build_formulas,
    search,
)
from factorylib.fractions import snap_value


def _parse_float_list(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split(",")], dtype=float)


def _parse_kv_float(s: str) -> tuple[str, float]:
    name, sep, value = s.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    return name, float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factorylib-endfield",
        description="Compute an optimal Wuling production plan and print it, "
        "along with any tied alternatives.",
    )
    parser.add_argument(
        "--base-supply",
        type=_parse_float_list,
        default=None,
        help="comma-separated 8-value resource supply ("
        + ", ".join(RESOURCE_NAMES)
        + "); default: 1.2e base",
    )
    parser.add_argument(
        "--max-forges", type=int, default=None, help="default: 12 (1.2e)"
    )
    parser.add_argument(
        "--metatransfer",
        type=_parse_float_list,
        action="append",
        default=None,
        dest="metatransfer",
        help="comma-separated 8-value metatransfer top-up; repeatable; "
        "default: the two standard metatransfers",
    )

    purify_building = parser.add_mutually_exclusive_group()
    purify_building.add_argument(
        "--purify-building", dest="purify_building", action="store_true", default=None
    )
    purify_building.add_argument(
        "--no-purify-building", dest="purify_building", action="store_false"
    )

    purify_node = parser.add_mutually_exclusive_group()
    purify_node.add_argument(
        "--purify-node", dest="purify_node", action="store_true", default=None
    )
    purify_node.add_argument(
        "--no-purify-node", dest="purify_node", action="store_false"
    )

    parser.add_argument(
        "--limit",
        type=_parse_kv_float,
        action="append",
        default=[],
        dest="limits",
        metavar="NAME=VALUE",
        help="override a formula's run-rate limit (repeatable), e.g. ya=0 to ban it",
    )
    parser.add_argument(
        "--formula-output",
        type=_parse_kv_float,
        action="append",
        default=[],
        dest="outputs",
        metavar="NAME=VALUE",
        help="override a formula's $/run output (repeatable)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="LP objective perturbation size used to find tied alternatives",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=4,
        help="max number of tied solutions to print (including the optimum)",
    )
    parser.add_argument(
        "--tie-tol",
        type=float,
        default=1e-6,
        help="$ tolerance for reporting a different discrete (z, metatransfer) "
        "branch as tied with the optimum",
    )
    return parser


def _fmt(x: float) -> str:
    return str(snap_value(x, warn=False))


def _format_result(label: str, result, formula_names: list[str]) -> str:
    lines = [
        f"{label}: dollar={_fmt(result.dollar_output)} ({result.dollar_output:.4f})"
    ]
    for name, rate in zip(formula_names, result.formula_rates):
        if abs(rate) > 1e-9:
            lines.append(f"    {name} = {_fmt(rate)} ({rate:.4f})")
    slack_parts = [
        f"{name}={_fmt(s)}"
        for name, s in zip(RESOURCE_NAMES, result.resource_slack)
        if abs(s) > 1e-9
    ]
    if slack_parts:
        lines.append("    slack: " + ", ".join(slack_parts))
    return "\n".join(lines)


def _build_config(args: argparse.Namespace) -> WulingConfig:
    kwargs: dict = {}
    if args.base_supply is not None:
        kwargs["base_supply"] = args.base_supply
    if args.max_forges is not None:
        kwargs["max_forges"] = args.max_forges
    if args.metatransfer is not None:
        kwargs["metatransfers"] = args.metatransfer
    if args.purify_building is not None:
        kwargs["purify_building"] = args.purify_building
    if args.purify_node is not None:
        kwargs["purify_node"] = args.purify_node
    if args.limits:
        kwargs["formula_limits"] = dict(args.limits)
    if args.outputs:
        kwargs["formula_outputs"] = dict(args.outputs)
    return WulingConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _build_config(args)

    best = search(config)
    print(_format_result("Optimal solution", best.result, best.formula_names))
    print(f"    z={best.z}, metatransfer={best.metatransfer.tolist()}")

    formulas = build_formulas(config)
    if not config.fix_hx_limit:
        formulas["hx"].limit = config.max_forges - best.z
    supply = config.base_supply + best.z * XI_PER_FORGE + best.metatransfer
    alt_result = find_alternatives(
        supply,
        list(formulas.values()),
        epsilon=args.epsilon,
        max_solutions=args.max_solutions,
    )
    if alt_result.alternatives:
        print("\nTied alternatives (same z/metatransfer, different LP vertex):")
        for i, alt in enumerate(alt_result.alternatives, 1):
            print(_format_result(f"  Alternative {i}", alt, best.formula_names))

    discrete_ties = [
        (result, z, mt)
        for result, z, mt in best.all_candidates
        if result.status == "optimal"
        and (z != best.z or not np.allclose(mt, best.metatransfer))
        and abs(result.dollar_output - best.result.dollar_output) <= args.tie_tol
    ]
    if discrete_ties:
        print("\nTied discrete branches (different z/metatransfer):")
        for result, z, mt in discrete_ties[: max(args.max_solutions - 1, 0)]:
            print(
                _format_result(
                    f"  z={z}, metatransfer={mt.tolist()}", result, best.formula_names
                )
            )

    return 0
