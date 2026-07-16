"""Sanity tests for pp_goals.py's tier-shape generators. These are pure
functions of a few numeric parameters with no dependency on the Wuling
formula set at all -- these tests build tiny standalone Formula lists
(not the real pp-scored formula set) so the tier *shapes* themselves can
be checked in isolation from the rest of the pp architecture.
"""

import numpy as np
import pytest

from factorylib.endfield.pp_goals import (
    hard_satisfaction_bonus,
    nonzero_production_tiers,
    satisfaction_tiers,
)
from factorylib.optimize import Formula, maximize_dollar


def _total_reward(tiers: list[tuple[float, float]], flow_supply: float) -> float:
    """Materialize (pp_per_unit, cap_width) tiers as 1-resource-per-unit
    Formula entries (simpler than pp_goals._materialize_tiers's
    input_per_multiple-scaled version, since that's an unrelated
    implementation detail of how it's wired into the real formula set --
    here we only care about the tier shape itself) and solve for the
    maximum reward achievable given flow_supply units of the underlying
    flow."""
    formulas = [
        Formula(consumption=np.array([1.0]), output=pp_per_unit, limit=cap_width)
        for pp_per_unit, cap_width in tiers
    ]
    result = maximize_dollar(np.array([flow_supply]), formulas)
    return result.dollar_output


def test_satisfaction_tiers_ramp_tiers_sum_to_target():
    """The first n_ramp_tiers tiers alone must exactly partition
    [0, target] -- reaching 100% should consume exactly target's worth
    of flow, not more or less."""
    target = 1090.0
    tiers = satisfaction_tiers(
        target, soft_cap_ratio=1.20, hard_cap_ratio=3.00, n_ramp_tiers=3
    )
    ramp_width_sum = sum(width for _, width in tiers[:3])
    assert ramp_width_sum == pytest.approx(target)


def test_satisfaction_tiers_total_width_caps_at_hard_cap_ratio():
    """Nothing should exist past target * hard_cap_ratio -- the combined
    width of every tier (ramp + soft + tail) must sum to exactly that,
    not infinity and not less."""
    target = 7000.0
    tiers = satisfaction_tiers(target, soft_cap_ratio=1.05, hard_cap_ratio=1.40)
    assert sum(width for _, width in tiers) == pytest.approx(target * 1.40)


def test_satisfaction_tiers_pp_per_unit_strictly_decreasing():
    """Concavity: each tier must be worth strictly less per unit than the
    last, so a plain $-maximizing LP naturally fills the steepest-slope
    tier first (see module docstring)."""
    tiers = satisfaction_tiers(1000.0, soft_cap_ratio=1.2, hard_cap_ratio=2.0)
    pps = [pp for pp, _ in tiers]
    assert all(a > b for a, b in zip(pps, pps[1:]))


def test_satisfaction_tiers_reward_plateaus_past_hard_cap():
    """The concrete sanity check: build a curve with soft_cap_ratio=105%,
    hard_cap_ratio=120%, then confirm the maximum achievable reward is
    identical whether 140% or 200% of target flow is available --
    overshooting the hard cap must never be worth anything further."""
    target = 100.0
    tiers = satisfaction_tiers(target, soft_cap_ratio=1.05, hard_cap_ratio=1.20)
    reward_140 = _total_reward(tiers, 1.4 * target)
    reward_200 = _total_reward(tiers, 2.0 * target)
    assert reward_140 == pytest.approx(reward_200)
    # and it must actually be less than what's achievable exactly at the
    # hard cap boundary would require unlimited flow to exceed further
    reward_at_hard_cap = _total_reward(tiers, 1.20 * target)
    assert reward_140 == pytest.approx(reward_at_hard_cap)


def test_satisfaction_tiers_reward_below_target_is_strictly_less():
    """Sanity check on the other side: reward at 50% of target must be
    strictly less than reward at 100%, confirming the ramp tiers
    actually pay out proportionally to progress."""
    target = 100.0
    tiers = satisfaction_tiers(target, soft_cap_ratio=1.2, hard_cap_ratio=2.0)
    reward_50 = _total_reward(tiers, 0.5 * target)
    reward_100 = _total_reward(tiers, target)
    assert reward_50 < reward_100


def test_nonzero_production_tiers_last_tier_is_unbounded():
    """Unlike satisfaction_tiers, nonzero_production_tiers has no real
    target to cap against -- the last tier must be unbounded (but still
    diminishing, i.e. worth less per unit than every prior tier)."""
    tiers = nonzero_production_tiers(n_tiers=3, first_cap=0.1, cap_growth=10.0)
    widths = [width for _, width in tiers]
    assert widths[-1] == np.inf
    assert all(np.isfinite(w) for w in widths[:-1])
    pps = [pp for pp, _ in tiers]
    assert all(a > b for a, b in zip(pps, pps[1:]))
    assert pps[-1] > 0.0  # "diminishing, but never worthless"


def test_nonzero_production_tiers_cap_widths_grow_geometrically():
    tiers = nonzero_production_tiers(n_tiers=3, first_cap=0.1, cap_growth=10.0)
    widths = [width for _, width in tiers[:-1]]
    assert widths == pytest.approx([0.1, 1.0])


def test_hard_satisfaction_bonus_returns_expected_shape():
    """(input_per_multiple, pp, limit) -- consumes exactly `target`,
    pays `bonus_pp`, and is capped at exactly 1 multiple (all-or-
    nothing: see factorylib.search's _integer_rates_valid, which is what
    actually enforces this when the formula is materialized with
    integer=True)."""
    input_per_multiple, pp, limit = hard_satisfaction_bonus(1090.0, 2000.0)
    assert input_per_multiple == 1090.0
    assert pp == 2000.0
    assert limit == 1.0
