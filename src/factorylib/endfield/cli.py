"""CLI: compute an optimal Wuling production plan and print it, along with
any tied alternatives (see factorylib.alternatives)."""

from __future__ import annotations

import argparse

import numpy as np

from factorylib.alternatives import find_alternatives
from factorylib.delivery import DeliverySimConfig
from factorylib.endfield.delivery import predict_delivery_selections
from factorylib.endfield.goals import WulingGoals
from factorylib.endfield.refine import refine
from factorylib.endfield.wuling import (
    FORMULA_LABELS,
    METATRANSFER_ITEMS,
    RESOURCE_LABELS,
    RESOURCE_NAMES,
    SECONDARY_GOAL_FORMULA_NAMES,
    XI_PER_FORGE,
    WulingConfig,
    build_formulas,
    search,
)
from factorylib.fractions import snap_value
from factorylib.optimize import OptimizeResult
from factorylib.search import SearchConfig


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
    parser.add_argument(
        "--stock-bill-cap",
        type=float,
        default=1090.0,
        help="max $/min actually sellable, limited by savings generation "
        "(default: 700 + 390 per the current Sky King Flats Construction "
        "Site + Cardiac Remediation Station rates)",
    )
    parser.add_argument(
        "--power-target",
        type=float,
        default=7000.0,
        help="W worth of batteries to aim for (default: 7000, average player demand)",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=2000,
        help="number of simulated-annealing moves to try when searching for "
        "a more-fit alternative to the optimal solution",
    )
    parser.add_argument(
        "--refine-seed",
        type=int,
        default=0,
        help="RNG seed for the refine search (reproducible by default)",
    )
    parser.add_argument(
        "--refine-backend",
        choices=["sa", "scipy"],
        default="sa",
        help="refine search backend (default: sa; see factorylib.endfield.refine)",
    )
    parser.add_argument(
        "--delivery-box-capacity",
        type=float,
        default=14000.0,
        help="items removed from the selected material per delivery job "
        "(default: 14000)",
    )
    parser.add_argument(
        "--delivery-jobs-per-day",
        type=int,
        default=2,
        help="delivery jobs run per day (default: 2)",
    )
    parser.add_argument(
        "--delivery-sim-days",
        type=int,
        default=100,
        help="number of days to simulate delivery-job material selection",
    )
    parser.add_argument(
        "--delivery-startup-days",
        type=float,
        default=1.0,
        help="days of accumulation before the first simulated delivery job",
    )
    return parser


def _fmt(x: float) -> str:
    return str(snap_value(x, warn=False))


def _format_metatransfer(mt: np.ndarray) -> str:
    """Metatransfer vectors are expressed in this module's internal
    resource-equivalent units, not the item you actually select in the
    game's Metatransfer menu (e.g. a nonzero "dop" entry literally means
    "select Dense Originium Powder") -- METATRANSFER_ITEMS names that
    item directly so the raw vector doesn't have to be reverse-engineered
    by hand."""
    parts = []
    for name, amount in zip(RESOURCE_NAMES, mt):
        if abs(amount) < 1e-9:
            continue
        item_name = METATRANSFER_ITEMS.get(name, RESOURCE_LABELS.get(name, name))
        parts.append(f"{_fmt(amount)} {item_name}")
    return "select " + ", ".join(parts) if parts else "none"


def _format_result(label: str, result, formula_names: list[str]) -> str:
    lines = [
        f"{label}: dollar={_fmt(result.dollar_output)} $/min "
        f"({result.dollar_output:.4f} $/min)"
    ]
    for name, rate in zip(formula_names, result.formula_rates):
        if abs(rate) > 1e-9:
            full_name = FORMULA_LABELS.get(name, name)
            lines.append(f"    {full_name}: {_fmt(rate)} multiples ({rate:.4f})")
    slack_parts = [
        f"{RESOURCE_LABELS.get(name, name)}={_fmt(s)}"
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
    print(f"    z={best.z}, metatransfer: {_format_metatransfer(best.metatransfer)}")

    formulas = build_formulas(config)
    if not config.fix_hx_limit:
        formulas["hx"].limit = config.max_forges - best.z
    supply = config.base_supply + best.z * XI_PER_FORGE + best.metatransfer

    # Exclude the zero-$ secondary-goal formulas from tie detection: any
    # slack they could (for free) absorb is a real LP degeneracy but not
    # an economically meaningful "tied solution" (see wuling.py's module
    # docstring).
    primary_names = [
        name for name in best.formula_names if name not in SECONDARY_GOAL_FORMULA_NAMES
    ]
    alt_result = find_alternatives(
        supply,
        [formulas[name] for name in primary_names],
        epsilon=args.epsilon,
        max_solutions=args.max_solutions,
    )
    if alt_result.alternatives:
        print("\nTied alternatives (same z/metatransfer, different LP vertex):")
        for i, alt in enumerate(alt_result.alternatives, 1):
            print(_format_result(f"  Alternative {i}", alt, primary_names))

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
                    f"  z={z}, metatransfer: {_format_metatransfer(mt)}",
                    result,
                    best.formula_names,
                )
            )

    goals = WulingGoals(
        stock_bill_cap=args.stock_bill_cap, power_target=args.power_target
    )
    search_config = SearchConfig(
        iterations=args.refine_iterations, seed=args.refine_seed
    )
    refined = refine(best, config, goals, search_config, backend=args.refine_backend)
    consumption = np.stack([f.consumption for f in formulas.values()], axis=1)
    refined_slack = np.maximum(0.0, supply - consumption @ refined.rates)
    refined_result = OptimizeResult(
        status="optimal",
        dollar_output=refined.dollar_output,
        formula_rates=refined.rates,
        resource_slack=refined_slack,
    )
    print(
        f"\nMost fit solution found (fitness={refined.fitness:.4f}, "
        f"backend={args.refine_backend}):"
    )
    print(_format_result("  Refined solution", refined_result, refined.formula_names))
    if refined.headroom_lost:
        lost = ", ".join(RESOURCE_LABELS.get(n, n) for n in refined.headroom_lost)
        print(
            f"  Warning: this solution fully saturates {lost}, which had spare "
            "capacity in the optimal solution above. This only checks the "
            "modeled resource balance, not physical topology (splitter "
            "wiring, priority overflow, backpressure) -- see "
            "factorylib.endfield.goals's module docstring."
        )

    delivery_config = DeliverySimConfig(
        startup_days=args.delivery_startup_days,
        simulation_days=args.delivery_sim_days,
        jobs_per_day=args.delivery_jobs_per_day,
        box_capacity=args.delivery_box_capacity,
    )
    rates_by_name = dict(zip(refined.formula_names, refined.rates))
    tally = predict_delivery_selections(rates_by_name, refined_slack, delivery_config)
    print(
        f"\nDelivery job prediction ({delivery_config.simulation_days} days, "
        f"{delivery_config.jobs_per_day} jobs/day, "
        f"{_fmt(delivery_config.box_capacity)}/job, after "
        f"{_fmt(delivery_config.startup_days)}-day startup): what a "
        "material's rate meets a target doesn't mean it's what gets picked --"
        " this simulates the depot's actual highest-amount auto-select."
    )
    if tally:
        for name, count in sorted(tally.items(), key=lambda kv: kv[1], reverse=True):
            if count > 0:
                print(f"    {name}: selected {count} times")
        never_selected = [name for name, count in tally.items() if count == 0]
        if never_selected:
            print(f"    (never selected: {', '.join(never_selected)})")
    else:
        print("    (nothing accumulates unconsumed in the depot)")

    return 0
