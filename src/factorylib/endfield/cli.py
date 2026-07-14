"""CLI: compute an optimal Wuling production plan and print it, along with
any tied alternatives (see factorylib.alternatives)."""

from __future__ import annotations

import argparse
import random

import numpy as np

from factorylib.alternatives import find_alternatives
from factorylib.delivery import DeliverySimConfig
from factorylib.endfield.delivery import predict_delivery_selections
from factorylib.endfield.goals import WulingGoals
from factorylib.endfield.refine import refine
from factorylib.endfield.wuling import (
    FORMULA_LABELS,
    GOOD_YIELD,
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
        "-b",
        "--base-supply",
        type=_parse_float_list,
        default=None,
        help="comma-separated 8-value resource supply ("
        + ", ".join(RESOURCE_NAMES)
        + "); default: 1.2e base",
    )
    parser.add_argument(
        "-f", "--max-forges", type=int, default=None, help="default: 12 (1.2e)"
    )
    parser.add_argument(
        "-m",
        "--metatransfer",
        type=_parse_float_list,
        action="append",
        default=None,
        dest="metatransfer",
        help="comma-separated 8-value metatransfer top-up; repeatable; "
        "default: the two standard metatransfers",
    )

    # No short flags on these four: they're store_true/store_false pairs
    # (both directions of one setting), and the single letters that would
    # naturally go with them ("p" for purify) are already claimed by
    # --power-target below.
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
        "-l",
        "--limit",
        type=_parse_kv_float,
        action="append",
        default=[],
        dest="limits",
        metavar="NAME=VALUE",
        help="override a formula's run-rate limit (repeatable), e.g. ya=0 to ban it",
    )
    parser.add_argument(
        "-o",
        "--formula-output",
        type=_parse_kv_float,
        action="append",
        default=[],
        dest="outputs",
        metavar="NAME=VALUE",
        help="override a formula's $/run output (repeatable)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1e-4,
        help="LP objective perturbation size used to find tied alternatives",
    )
    parser.add_argument(
        "-n",
        "--max-solutions",
        type=int,
        default=4,
        help="max number of tied solutions to print (including the optimum)",
    )
    parser.add_argument(
        "-t",
        "--tie-tol",
        type=float,
        default=1e-6,
        help="$ tolerance for reporting a different discrete (z, metatransfer) "
        "branch as tied with the optimum",
    )
    parser.add_argument(
        "-c",
        "--stock-bill-cap",
        type=float,
        default=1090.0,
        help="max $/min actually sellable, limited by savings generation "
        "(default: 700 + 390 per the current Sky King Flats Construction "
        "Site + Cardiac Remediation Station rates)",
    )
    parser.add_argument(
        "-p",
        "--power-target",
        type=float,
        default=7000.0,
        help="W worth of batteries to aim for (default: 7000, average player demand)",
    )
    parser.add_argument(
        "-w",
        "--complexity-weight",
        type=float,
        default=1.0,
        help="weight of the simplicity/denominator penalty in the refine search "
        "(default: 1.0; lower values trade simpler fractions for more $/min -- "
        "e.g. on 1.2e full, 0.01 recovers ~99%% of LP-optimal $ vs ~91%% at 1.0, "
        "at the cost of much larger denominators)",
    )
    parser.add_argument(
        "-i",
        "--refine-iterations",
        type=int,
        default=2000,
        help="number of simulated-annealing moves to try when searching for "
        "a more-fit alternative to the optimal solution",
    )
    parser.add_argument(
        "-s",
        "--refine-seed",
        type=int,
        default=0,
        help="RNG seed for the refine search (reproducible by default; "
        "ignored if -R/--random-seed is given)",
    )
    parser.add_argument(
        "-R",
        "--random-seed",
        action="store_true",
        help="use a random seed for the refine search instead of -s/--refine-seed "
        "(the actual seed used is printed, so the result can be reproduced later)",
    )
    parser.add_argument(
        "-r",
        "--refine-backend",
        choices=["sa", "scipy"],
        default="sa",
        help="refine search backend (default: sa; see factorylib.endfield.refine)",
    )
    parser.add_argument(
        "-B",
        "--delivery-box-capacity",
        type=float,
        default=14000.0,
        help="items removed from the selected material per delivery job "
        "(default: 14000)",
    )
    parser.add_argument(
        "-j",
        "--delivery-jobs-per-day",
        type=int,
        default=2,
        help="delivery jobs run per day (default: 2)",
    )
    parser.add_argument(
        "-d",
        "--delivery-sim-days",
        type=int,
        default=100,
        help="number of days to simulate delivery-job material selection",
    )
    parser.add_argument(
        "-u",
        "--delivery-startup-days",
        type=float,
        default=1.0,
        help="days of accumulation before the first simulated delivery job",
    )
    parser.add_argument(
        "-C",
        "--delivery-depot-capacity",
        type=float,
        default=80_000.0,
        help="max amount of any one material the depot can hold (default: 80000)",
    )
    parser.add_argument(
        "--delivery-seed",
        type=int,
        default=0,
        help="RNG seed for delivery-job tie-breaking (reproducible by default)",
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


def _format_income_breakdown(
    result, formula_names: list[str], formulas: dict, stock_bill_cap: float
) -> str:
    """Breaks down $/min income by sellable good: absolute $/min, % of
    total produced, and % of the stock-bill-cap goal -- plus the overall
    total as a % of that goal, so it's immediately clear how it compares
    without doing the division by hand."""
    lines = []
    total = result.dollar_output
    if stock_bill_cap > 1e-9:
        lines.append(
            f"    {100 * total / stock_bill_cap:.1f}% of {_fmt(stock_bill_cap)} "
            "$/min goal"
        )
    contributions = [
        (FORMULA_LABELS.get(name, name), rate * formulas[name].output)
        for name, rate in zip(formula_names, result.formula_rates)
        if name in formulas and rate * formulas[name].output > 1e-9
    ]
    if contributions:
        lines.append("    Income breakdown:")
        for label, amount in sorted(contributions, key=lambda kv: kv[1], reverse=True):
            pct_of_produced = 100 * amount / total if total > 1e-9 else 0.0
            pct_of_goal = (
                100 * amount / stock_bill_cap if stock_bill_cap > 1e-9 else 0.0
            )
            lines.append(
                f"        {label}: {_fmt(amount)} $/min "
                f"({pct_of_produced:.1f}% of produced, {pct_of_goal:.1f}% of goal)"
            )
    return "\n".join(lines)


def _format_result(
    label: str,
    result,
    formula_names: list[str],
    *,
    formulas: dict | None = None,
    stock_bill_cap: float | None = None,
) -> str:
    lines = [
        f"{label}: dollar={_fmt(result.dollar_output)} $/min "
        f"({result.dollar_output:.4f} $/min)"
    ]
    if formulas is not None and stock_bill_cap is not None:
        breakdown = _format_income_breakdown(
            result, formula_names, formulas, stock_bill_cap
        )
        if breakdown:
            lines.append(breakdown)
    for name, rate in zip(formula_names, result.formula_rates):
        if abs(rate) > 1e-9:
            full_name = FORMULA_LABELS.get(name, name)
            note = (
                " (net -- nothing else in this model consumes it)"
                if name in SECONDARY_GOAL_FORMULA_NAMES
                else ""
            )
            item_yield = GOOD_YIELD.get(name)
            item_rate = f" = {_fmt(rate * item_yield)}/min" if item_yield else ""
            lines.append(
                f"    {full_name}: {_fmt(rate)} multiples ({rate:.4f}){item_rate}{note}"
            )
    slack_parts = [
        f"{RESOURCE_LABELS.get(name, name)}={_fmt(s)}"
        for name, s in zip(RESOURCE_NAMES, result.resource_slack)
        if abs(s) > 1e-9
    ]
    if slack_parts:
        lines.append("    slack: " + ", ".join(slack_parts))
    return "\n".join(lines)


def _format_forge_allocation(z: int, max_forges: int) -> str:
    """Spells out what "z" means physically: z forges feed the Xiranite
    supply directly, the rest cap how much Heavy Xiranite can be made."""
    hx_forges = max_forges - z
    return (
        f"    Forge of the Sky: {max_forges} total -- "
        f"{z} -> Xiranite supply (+{_fmt(z * 30)}/min Xiranite), "
        f"{hx_forges} -> Heavy Xiranite capacity (max {hx_forges} multiples)"
    )


def _format_material_balance(
    supply: np.ndarray, rates_by_name: dict[str, float], formulas: dict
) -> str:
    """Per-resource source/sink breakdown: how much comes from the base
    supply (mining/Forge of the Sky/Metatransfer) plus which formulas
    produce or consume it, and the net surplus -- answers "is this rate
    inclusive of other consumers, or the excess" directly, instead of
    requiring it to be reconstructed by hand from the formula list."""
    lines = ["  Material balance (net = total produced - total consumed):"]
    for k, resource_name in enumerate(RESOURCE_NAMES):
        sources = []
        sinks = []
        if abs(supply[k]) > 1e-9:
            sources.append(
                ("base supply (mining/Forge of the Sky/Metatransfer)", supply[k])
            )
        for name, rate in rates_by_name.items():
            if abs(rate) < 1e-9 or name not in formulas:
                continue
            flow = rate * formulas[name].consumption[k]
            label = FORMULA_LABELS.get(name, name)
            if flow > 1e-9:
                sinks.append((label, flow))
            elif flow < -1e-9:
                sources.append((label, -flow))
        if not sources and not sinks:
            continue
        net = sum(v for _, v in sources) - sum(v for _, v in sinks)
        resource_label = RESOURCE_LABELS.get(resource_name, resource_name)
        lines.append(f"    {resource_label}:")
        for label, v in sources:
            lines.append(f"        +{_fmt(v)}/min from {label}")
        for label, v in sinks:
            lines.append(f"        -{_fmt(v)}/min to {label}")
        lines.append(f"        net: {_fmt(net)}/min")
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
    formulas = build_formulas(config)
    if not config.fix_hx_limit:
        formulas["hx"].limit = config.max_forges - best.z
    print(
        _format_result(
            "Optimal solution",
            best.result,
            best.formula_names,
            formulas=formulas,
            stock_bill_cap=args.stock_bill_cap,
        )
    )
    print(f"    z={best.z}, metatransfer: {_format_metatransfer(best.metatransfer)}")
    print(_format_forge_allocation(best.z, config.max_forges))

    supply = config.base_supply + best.z * XI_PER_FORGE + best.metatransfer
    print(
        _format_material_balance(
            supply, dict(zip(best.formula_names, best.result.formula_rates)), formulas
        )
    )

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
        stock_bill_cap=args.stock_bill_cap,
        power_target=args.power_target,
        complexity_weight=args.complexity_weight,
    )
    refine_seed = random.randint(0, 2**31 - 1) if args.random_seed else args.refine_seed
    if args.random_seed:
        print(
            f"\n(using random refine seed: {refine_seed} -- "
            f"pass -s {refine_seed} to reproduce)"
        )
    search_config = SearchConfig(iterations=args.refine_iterations, seed=refine_seed)
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
    print(
        _format_result(
            "  Refined solution",
            refined_result,
            refined.formula_names,
            formulas=formulas,
            stock_bill_cap=args.stock_bill_cap,
        )
    )
    print(
        _format_material_balance(
            supply, dict(zip(refined.formula_names, refined.rates)), formulas
        )
    )
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
        depot_capacity=args.delivery_depot_capacity,
        seed=args.delivery_seed,
    )
    rates_by_name = dict(zip(refined.formula_names, refined.rates))
    tally = predict_delivery_selections(rates_by_name, refined_slack, delivery_config)
    print(
        f"\nDelivery job prediction ({delivery_config.simulation_days} days, "
        f"{delivery_config.jobs_per_day} jobs/day, "
        f"{_fmt(delivery_config.box_capacity)}/job, depot cap "
        f"{_fmt(delivery_config.depot_capacity)}, after "
        f"{_fmt(delivery_config.startup_days)}-day startup): what a "
        "material's rate meets a target doesn't mean it's what gets picked --"
        " this simulates the depot's actual highest-amount auto-select "
        "(ties broken randomly)."
    )
    if tally:
        total_jobs = sum(tally.values())
        for name, count in sorted(tally.items(), key=lambda kv: kv[1], reverse=True):
            if count > 0:
                pct = 100 * count / total_jobs if total_jobs > 0 else 0.0
                print(f"    {name}: selected {count} times ({pct:.1f}%)")
        never_selected = [name for name, count in tally.items() if count == 0]
        if never_selected:
            print(f"    (never selected: {', '.join(never_selected)})")
    else:
        print("    (nothing accumulates unconsumed in the depot)")

    return 0
