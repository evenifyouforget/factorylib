"""CLI: compute an optimal Wuling production plan and print it, along with
any tied alternatives (see factorylib.alternatives)."""

from __future__ import annotations

import argparse
import random

import numpy as np

from factorylib.alternatives import find_alternatives
from factorylib.delivery import DeliverySimConfig, simulate_delivery_selections
from factorylib.endfield.delivery import accumulation_rates
from factorylib.endfield.diagram import generate_diagram
from factorylib.endfield.goals import item_rates
from factorylib.endfield.pp_goals import (
    PPGoals,
    build_pp_formulas,
    is_pp_bookkeeping_formula,
    pp_supply,
)
from factorylib.endfield.refine import refine
from factorylib.endfield.wuling import (
    FORMULA_LABELS,
    GOOD_YIELD,
    METATRANSFER_ITEMS,
    POWER_YIELD,
    RESOURCE_LABELS,
    RESOURCE_NAMES,
    SECONDARY_GOAL_FORMULA_NAMES,
    SECONDARY_PLUMBING_FORMULA_NAMES,
    SELL_PRIORITY,
    WulingConfig,
    build_formulas,
    full_supply,
    search,
)
from factorylib.fractions import snap_value
from factorylib.optimize import OptimizeResult
from factorylib.priority_sell import allocate_by_priority
from factorylib.search import SearchConfig

# Gear Components have no real "100%" target in the pp system (a Nonzero
# Production Goal -- see pp_goals.nonzero_production_tiers): this is
# purely an informational reference for display, not tied to any pp
# mechanism, matching the spec's own framing ("even 0.5/min of Cuprium
# Component is already ample").
_GEAR_MIN_TARGET_REFERENCE = 0.5


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
        help=f"comma-separated {len(RESOURCE_NAMES)}-value resource supply ("
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
        help=f"comma-separated {len(RESOURCE_NAMES)}-value metatransfer top-up; "
        "repeatable; default: the two standard metatransfers",
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
        default=0.1,
        help="weight of the simplicity/denominator penalty in the refine search "
        "(default: 0.1; lower values trade simpler fractions for more $/min and "
        "more reliably satisfying power/delivery/gear goals; higher values "
        "prioritize simple fractions over those goals)",
    )
    parser.add_argument(
        "-i",
        "--refine-iterations",
        type=int,
        default=6000,
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
    parser.add_argument(
        "--diagram",
        type=str,
        default=None,
        metavar="PATH",
        help="write a Graphviz diagram of the refined solution's active formulas "
        "to PATH (format inferred from its extension, e.g. plan.png); skipped "
        "with a notice if the graphviz package or its `dot` executable isn't "
        "installed",
    )
    return parser


def _fmt(x: float) -> str:
    return str(snap_value(x, warn=False))


def _fmt_pair(x: float) -> str:
    """ "[exact fraction = decimal]" -- one consistent notation for every
    place a rate/dollar amount is shown both ways, instead of each call
    site inventing its own "X (Y)" layout."""
    return f"[{_fmt(x)} = {x:.4f}]"


def _pct(rate: float, target: float, target_label: str) -> str:
    """ "X% of <target_label> goal", or a no-requirement note if target<=0.
    target_label is pre-formatted by the caller (e.g. "7000 W",
    "15/min") so the unit's spacing/placement is exactly right either
    way."""
    if target <= 0:
        return "no goal set"
    return f"{100 * rate / target:.1f}% of {target_label} goal"


def _goal_shortfall_warnings(
    rate: float, target: float, unit: str, goal_name: str
) -> list[str]:
    """A single "Warning: ..." line if rate falls short of target (empty
    list otherwise, or if target<=0 meaning "not a real goal"). Surfaces
    gaps like "power target never actually hit" that would otherwise be
    silently absent from the report -- see the CLI's own history of that
    exact confusion. unit is glued directly to numbers with no space if
    it starts with "/" (e.g. "/min"), space-separated otherwise (e.g. "W").
    """
    if target <= 0 or rate >= target - 1e-9:
        return []
    pct = 100 * rate / target
    sep = "" if unit.startswith("/") else " "
    return [
        f"  Warning: only {pct:.1f}% of the {_fmt(target)}{sep}{unit} {goal_name} "
        f"goal is met ({_fmt(rate)}{sep}{unit} produced)"
    ]


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


def _dollar_contributions(
    result, formula_names: list[str], formulas: dict
) -> dict[str, float]:
    """Every formula's $/min contribution, keyed by internal formula name
    (not label) -- shared by the income breakdown display and by the
    delivery-job wiring, which needs the unsold portion (see
    allocate_by_priority) as an accumulating-material rate."""
    return {
        name: rate * formulas[name].output
        for name, rate in zip(formula_names, result.formula_rates)
        if name in formulas and rate * formulas[name].output > 1e-9
    }


def _format_income_breakdown(
    result, formula_names: list[str], formulas: dict, stock_bill_cap: float
) -> str:
    """Breaks down $/min income by sellable good: absolute $/min, % of
    total produced, % of the stock-bill-cap goal, and how much of it
    actually gets sold vs. accumulates unsold. The outpost's $ savings
    only regenerate at stock_bill_cap, so goods are sold in a fixed
    priority order (SELL_PRIORITY), not proportionally to production --
    once the cap is exhausted, lower-priority goods simply pile up
    unsold instead of being sold at a discount (see
    factorylib.priority_sell)."""
    lines = []
    total = result.dollar_output
    if stock_bill_cap > 1e-9:
        lines.append(
            f"    {100 * total / stock_bill_cap:.1f}% of {_fmt(stock_bill_cap)} "
            "$/min goal"
        )
    contributions = _dollar_contributions(result, formula_names, formulas)
    if contributions:
        sold, unsold = allocate_by_priority(
            contributions, list(SELL_PRIORITY), stock_bill_cap
        )
        lines.append("    Income breakdown:")
        for name, amount in sorted(
            contributions.items(), key=lambda kv: kv[1], reverse=True
        ):
            label = FORMULA_LABELS.get(name, name)
            pct_of_produced = 100 * amount / total if total > 1e-9 else 0.0
            pct_of_goal = (
                100 * amount / stock_bill_cap if stock_bill_cap > 1e-9 else 0.0
            )
            note = ""
            if unsold.get(name, 0.0) > 1e-9:
                note = (
                    f" -- sold: {_fmt(sold[name])} $/min, "
                    f"accumulating: {_fmt(unsold[name])} $/min"
                )
            lines.append(
                f"        {label}: {_fmt(amount)} $/min "
                f"({pct_of_produced:.1f}% of produced, {pct_of_goal:.1f}% of goal)"
                f"{note}"
            )
        total_unsold = sum(unsold.values())
        if total_unsold > 1e-9:
            lines.append(
                f"        (outpost savings only regenerate at "
                f"{_fmt(stock_bill_cap)} $/min -- {_fmt(total_unsold)} $/min "
                "worth of goods above can't be sold yet and piles up "
                "physically instead)"
            )
    return "\n".join(lines)


def _format_result(
    label: str,
    result,
    formula_names: list[str],
    *,
    formulas: dict | None = None,
    stock_bill_cap: float | None = None,
    fitness_value: float | None = None,
) -> str:
    header = f"{label}: dollar={_fmt_pair(result.dollar_output)} $/min"
    if fitness_value is not None:
        header += f", fitness={fitness_value:.4f}"
    lines = [header]
    if formulas is not None and stock_bill_cap is not None:
        breakdown = _format_income_breakdown(
            result, formula_names, formulas, stock_bill_cap
        )
        if breakdown:
            lines.append(breakdown)
    for name, rate in zip(formula_names, result.formula_rates):
        if abs(rate) > 1e-9 and not is_pp_bookkeeping_formula(name):
            full_name = FORMULA_LABELS.get(name, name)
            item_yield = GOOD_YIELD.get(name)
            power_yield = POWER_YIELD.get(name)
            if item_yield:
                extra = f" = {_fmt_pair(rate * item_yield)}/min"
            elif power_yield:
                extra = f" = {_fmt_pair(rate * power_yield)} W"
            elif name in SECONDARY_GOAL_FORMULA_NAMES and name != "sandleaf_powder":
                extra = " (net -- nothing else in this model consumes it)"
            else:
                extra = ""
            lines.append(f"    {full_name}: {_fmt_pair(rate)} multiples{extra}")
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
    pp_goals = PPGoals(
        dollar_target=args.stock_bill_cap,
        power_target=args.power_target,
        complexity_weight=args.complexity_weight,
        delivery_box_capacity=args.delivery_box_capacity,
        delivery_jobs_per_day=float(args.delivery_jobs_per_day),
    )

    best = search(config)
    formulas = build_formulas(config)
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

    supply = full_supply(config)
    print(
        _format_material_balance(
            supply, dict(zip(best.formula_names, best.result.formula_rates)), formulas
        )
    )

    # Exclude the zero-$ secondary-goal formulas (and the plumbing that
    # solely feeds them) from the tie-detection LP entirely: any slack
    # they could (for free) absorb is a real LP degeneracy but not an
    # economically meaningful "tied solution" (see wuling.py's module
    # docstring), and leaving their own internal degeneracy in the LP
    # risks a *different* HiGHS vertex for one of them (even with an
    # unrelated formula perturbed) tripping the structurally-distinct
    # check below as a false positive. sandleaf_powder is deliberately
    # kept in despite being in SECONDARY_GOAL_FORMULA_NAMES: ori_to_dop
    # (core) genuinely needs its output, so excluding it would starve
    # ori_to_dop and corrupt the *baseline* dollar figure within this
    # filtered sub-problem, not just suppress a spurious alternative.
    _tie_detection_exclude = tuple(
        name
        for name in SECONDARY_GOAL_FORMULA_NAMES + SECONDARY_PLUMBING_FORMULA_NAMES
        if name != "sandleaf_powder"
    )
    primary_names = [
        name for name in best.formula_names if name not in _tie_detection_exclude
    ]
    primary_formulas = [formulas[name] for name in primary_names]
    # Perturb $-bearing formulas' outputs (Part 2's own "adjusting
    # weights" framing implies a zero-$ formula has no weight to begin
    # with) plus the forge/metatransfer choice formulas (integer=True,
    # zero $ output but still a genuine discrete choice worth finding
    # ties between -- see wuling.py::search's docstring). Restricting to
    # these isn't required for correctness -- find_alternatives' own
    # dollar-closeness check catches a bad alternative regardless of
    # which direction produced it -- but it means fewer wasted solves get
    # discarded by that check in the first place.
    directions = [
        vec
        for vec, name in zip(np.eye(len(primary_formulas)), primary_names)
        if formulas[name].output > 0 or formulas[name].integer
    ]
    alt_result = find_alternatives(
        supply,
        primary_formulas,
        epsilon=args.epsilon,
        max_solutions=args.max_solutions,
        directions=directions,
    )
    if alt_result.alternatives:
        print("\nTied alternatives (same $, different plan -- may include a")
        print("different forge allocation or Metatransfer choice):")
        for i, alt in enumerate(alt_result.alternatives, 1):
            print(_format_result(f"  Alternative {i}", alt, primary_names))

    refine_seed = random.randint(0, 2**31 - 1) if args.random_seed else args.refine_seed
    if args.random_seed:
        print(
            f"\n(using random refine seed: {refine_seed} -- "
            f"pass -s {refine_seed} to reproduce)"
        )
    search_config = SearchConfig(iterations=args.refine_iterations, seed=refine_seed)
    refined = refine(best, config, pp_goals, search_config, backend=args.refine_backend)
    # The refined search runs over the FULL pp-scored formula set (real
    # recipes + pp-tier/bonus/delivery-quota bookkeeping -- see
    # pp_goals.build_pp_formulas), a superset of `formulas`/`supply`
    # above, so material balance/slack need their own extended versions.
    pp_formulas = build_pp_formulas(config, pp_goals)
    pp_supply_arr = pp_supply(config)
    pp_consumption = np.stack(
        [pp_formulas[name].consumption for name in refined.formula_names], axis=1
    )
    refined_slack = np.maximum(0.0, pp_supply_arr - pp_consumption @ refined.rates)
    refined_result = OptimizeResult(
        status="optimal",
        dollar_output=refined.dollar_output,
        formula_rates=refined.rates,
        resource_slack=refined_slack,
    )
    print(f"\nMost fit solution found (backend={args.refine_backend}):")
    print(
        _format_result(
            "  Refined solution",
            refined_result,
            refined.formula_names,
            formulas=formulas,
            stock_bill_cap=args.stock_bill_cap,
            fitness_value=refined.fitness,
        )
    )
    print(f"  Prosperity Points: {_fmt_pair(refined.pp_output)}")
    print(
        _format_material_balance(
            pp_supply_arr, dict(zip(refined.formula_names, refined.rates)), pp_formulas
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

    rates_by_name = dict(zip(refined.formula_names, refined.rates))
    power_rate = sum(
        POWER_YIELD.get(name, 0.0) * rate for name, rate in rates_by_name.items()
    )
    good_rates = item_rates(rates_by_name)
    power_target_label = f"{_fmt(pp_goals.power_target)} W"
    power_pct = _pct(power_rate, pp_goals.power_target, power_target_label)
    print(f"  Power: {_fmt_pair(power_rate)} W ({power_pct})")

    # Delivery Job Quota achieved: sum of every delivery_quota_from_*
    # formula's rate (see pp_goals module docstring) -- how many boxes'
    # worth of distinct materials this plan can supply per day, against
    # the pp_goals.delivery_jobs_per_day target.
    quota_contributions = {
        name: rate
        for name, rate in rates_by_name.items()
        if name.startswith("delivery_quota_from_") and rate > 1e-9
    }
    quota_achieved = sum(quota_contributions.values())
    quota_target_label = f"{_fmt(pp_goals.delivery_jobs_per_day)} jobs/day"
    quota_pct = _pct(quota_achieved, pp_goals.delivery_jobs_per_day, quota_target_label)
    print(f"  Delivery quota: {_fmt_pair(quota_achieved)} ({quota_pct})")
    for name, rate in sorted(
        quota_contributions.items(), key=lambda kv: kv[1], reverse=True
    ):
        resource_name = name.removeprefix("delivery_quota_from_")
        label = RESOURCE_LABELS.get(resource_name, resource_name)
        print(f"      {label}: {_fmt_pair(rate)} quota")

    for warning in (
        _goal_shortfall_warnings(
            refined.dollar_output, pp_goals.dollar_target, "$/min", "stock-bill"
        )
        + _goal_shortfall_warnings(power_rate, pp_goals.power_target, "W", "power")
        + _goal_shortfall_warnings(
            quota_achieved, pp_goals.delivery_jobs_per_day, "jobs/day", "delivery quota"
        )
    ):
        print(warning)

    # Gear Components have no real "100%" target in the pp system (a
    # Nonzero Production Goal -- see pp_goals.nonzero_production_tiers):
    # _GEAR_MIN_TARGET_REFERENCE is purely informational, not tied to any
    # pp mechanism, just the spec's own framing ("even 0.5/min of
    # Cuprium Component is already ample") for gauging "is this enough".
    for good_name in (
        "hetonite_component",
        "xiranite_component",
        "cuprium_component",
        "ferrium_component",
    ):
        label = FORMULA_LABELS.get(good_name, good_name)
        rate = good_rates.get(good_name, 0.0)
        target_label = f"{_fmt(_GEAR_MIN_TARGET_REFERENCE)}/min"
        pct = _pct(rate, _GEAR_MIN_TARGET_REFERENCE, target_label)
        print(f"  {label}: {_fmt_pair(rate)}/min ({pct})")
        for warning in _goal_shortfall_warnings(
            rate, _GEAR_MIN_TARGET_REFERENCE, "/min", f"{label} gear"
        ):
            print(warning)

    delivery_config = DeliverySimConfig(
        startup_days=args.delivery_startup_days,
        simulation_days=args.delivery_sim_days,
        jobs_per_day=args.delivery_jobs_per_day,
        box_capacity=args.delivery_box_capacity,
        depot_capacity=args.delivery_depot_capacity,
        seed=args.delivery_seed,
    )
    accumulating = accumulation_rates(rates_by_name, refined_slack)
    # Goods the outpost's $ savings can't currently afford to buy (see
    # _format_income_breakdown) don't vanish -- they pile up physically,
    # so they're delivery-job candidates too, on top of the base-resource
    # slack and unconsumed secondary goods accumulation_rates() already
    # covers.
    refined_contributions = _dollar_contributions(
        refined_result, refined.formula_names, formulas
    )
    refined_sold, refined_unsold = allocate_by_priority(
        refined_contributions, list(SELL_PRIORITY), args.stock_bill_cap
    )
    for name, unsold_dollar in refined_unsold.items():
        good_yield = GOOD_YIELD.get(name)
        output = formulas[name].output
        if unsold_dollar <= 1e-9 or not good_yield or not output:
            continue
        label = FORMULA_LABELS.get(name, name)
        accumulating[label] = (
            accumulating.get(label, 0.0) + unsold_dollar * good_yield / output
        )
    delivery_result = simulate_delivery_selections(accumulating, delivery_config)
    tally = delivery_result.tally
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
        total_jobs = sum(tally.values()) + delivery_result.failed_jobs
        for name, count in sorted(tally.items(), key=lambda kv: kv[1], reverse=True):
            if count > 0:
                pct = 100 * count / total_jobs if total_jobs > 0 else 0.0
                print(f"    {name}: selected {count} times ({pct:.1f}%)")
        never_selected = [name for name, count in tally.items() if count == 0]
        if never_selected:
            print(f"    (never selected: {', '.join(never_selected)})")
    else:
        print("    (nothing accumulates unconsumed in the depot)")
    if delivery_result.failed_jobs > 0:
        total_jobs = sum(tally.values()) + delivery_result.failed_jobs
        pct = 100 * delivery_result.failed_jobs / total_jobs
        print(
            f"  Warning: failed to pack goods due to insufficient materials on "
            f"{delivery_result.failed_jobs} of {total_jobs} simulated jobs "
            f"({pct:.1f}%) -- no material had accumulated the "
            f"{_fmt(delivery_config.box_capacity)} needed to fill a box"
        )

    if args.diagram:
        written = generate_diagram(
            rates_by_name,
            formulas,
            RESOURCE_NAMES,
            RESOURCE_LABELS,
            FORMULA_LABELS,
            args.diagram,
        )
        if written:
            print(f"\nDiagram written to {written}")
        else:
            print(
                f"\n(skipped diagram: install the graphviz package and its `dot` "
                f"executable to write one to {args.diagram})"
            )

    return 0
