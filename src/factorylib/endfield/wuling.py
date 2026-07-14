"""Standard Wuling environment: configurable formula set + forge/metatransfer
search, generalizing the "1.2e full" model (tests/wuling/test_wuling_1p2e.py).

Resource vector (8): [xi, ori, ferr, cup_ore, cup, sew, eff, inert]
Formula order: cup_conv xi_sew sc lc hp hx ya yc xi_sell cp_sell [purify]
               [purify_node]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factorylib.optimize import Formula, OptimizeResult, maximize_dollar

RESOURCE_NAMES = ["xi", "ori", "ferr", "cup_ore", "cup", "sew", "eff", "inert"]
FORMULA_NAMES = [
    "cup_conv",
    "xi_sew",
    "sc",
    "lc",
    "hp",
    "hx",
    "ya",
    "yc",
    "xi_sell",
    "cp_sell",
    "purify",
    "purify_node",
]

DEFAULT_BASE_SUPPLY = (0, 540, 90, 240, 0, 0, 0, 0)  # 1.2e base
DEFAULT_MAX_FORGES = 12
DEFAULT_METATRANSFERS = (
    (0, 50, 0, 0, 0, 0, 0, 0),
    (0, 0, 25, 0, 0, 0, 0, 0),
)
XI_PER_FORGE = np.array([30, 0, 0, 0, 0, 0, 0, 0], dtype=float)


@dataclass
class WulingConfig:
    """Standard Wuling environment configuration.

    Defaults reproduce "1.2e full" (everything on) exactly.

    Args:
        base_supply: length-8 resource supply vector (before forge/metatransfer
            top-ups), in RESOURCE_NAMES order.
        max_forges: number of Forge of the Sky units to split between
            Xiranite supply (z) and Heavy Xiranite capacity (max_forges - z).
        metatransfers: alternate resource top-up choices searched alongside z.
        purify_building: whether the Purification Building formula (inert ->
            eff) is included.
        purify_node: whether the Test Area Purification Node formula
            (sew -> eff, max 12 multiples) is included.
        formula_limits: per-formula `limit` overrides, e.g. {"ya": 0} to ban
            a formula. Keys must be in FORMULA_NAMES.
        formula_outputs: per-formula `output` ($/run) overrides. Keys must be
            in FORMULA_NAMES.
        fix_hx_limit: if True, search() will not override formulas["hx"].limit
            each z iteration (use the configured/overridden limit as-is).
    """

    base_supply: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_BASE_SUPPLY, dtype=float)
    )
    max_forges: int = DEFAULT_MAX_FORGES
    metatransfers: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array(mt, dtype=float) for mt in DEFAULT_METATRANSFERS
        ]
    )
    purify_building: bool = True
    purify_node: bool = True
    formula_limits: dict[str, float] = field(default_factory=dict)
    formula_outputs: dict[str, float] = field(default_factory=dict)
    fix_hx_limit: bool = False

    def __post_init__(self) -> None:
        self.base_supply = np.asarray(self.base_supply, dtype=float)
        if self.base_supply.shape != (8,):
            raise ValueError(
                "base_supply must be length 8: " + ", ".join(RESOURCE_NAMES)
            )
        self.metatransfers = [np.asarray(mt, dtype=float) for mt in self.metatransfers]
        for name in self.formula_limits:
            if name not in FORMULA_NAMES:
                raise ValueError(f"Unknown formula name in formula_limits: {name!r}")
        for name in self.formula_outputs:
            if name not in FORMULA_NAMES:
                raise ValueError(f"Unknown formula name in formula_outputs: {name!r}")


def build_formulas(config: WulingConfig) -> dict[str, Formula]:
    """Build a fresh dict of 1.2e-model Formulas per config.

    Ports tests/wuling/test_wuling_1p2e.py::_make_1p2e_formulas verbatim,
    then applies config.formula_limits / config.formula_outputs overrides.
    """
    f = {
        # 30 cup_ore -> 30 cup + 30 sew
        "cup_conv": Formula(
            consumption=np.array([0, 0, 0, 30, -30, -30, 0, 0], dtype=float), output=0
        ),
        # 30 xi + 30 sew -> 30 eff + 30 inert
        "xi_sew": Formula(
            consumption=np.array([30, 0, 0, 0, 0, 30, -30, -30], dtype=float), output=0
        ),
        # SC: (60 eff + 30 ferr -> 30 Xircon + 30 sew) + (30 Xircon + 240 ori -> 6 SC)
        "sc": Formula(
            consumption=np.array([0, 240, 30, 0, 0, -30, 60, 0], dtype=float),
            output=54 * 6,
        ),
        # LC: 30 xi + 180 ori -> 6 LC
        "lc": Formula(
            consumption=np.array([30, 180, 0, 0, 0, 0, 0, 0], dtype=float),
            output=25 * 6,
        ),
        # HP: (240 cup + 30 ferr -> 30 Hetonite + 30 sew) + (30 Hetonite -> 6 HP)
        "hp": Formula(
            consumption=np.array([0, 0, 30, 0, 240, -30, 0, 0], dtype=float),
            output=48 * 6,
        ),
        # HX: 60 xi + 30 eff -> 6 HX
        "hx": Formula(
            consumption=np.array([60, 0, 0, 0, 0, 0, 30, 0], dtype=float),
            output=27 * 6,
        ),
        # YA: 120 cup -> 6 ya
        "ya": Formula(
            consumption=np.array([0, 0, 0, 0, 120, 0, 0, 0], dtype=float),
            output=22 * 6,
        ),
        # YC: 120 ferr -> 6 yc
        "yc": Formula(
            consumption=np.array([0, 0, 120, 0, 0, 0, 0, 0], dtype=float),
            output=16 * 6,
        ),
        # Sell xi at $1
        "xi_sell": Formula(
            consumption=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float), output=1
        ),
        # Sell cup at $1 (after cup_conv conversion)
        "cp_sell": Formula(
            consumption=np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float), output=1
        ),
    }
    if config.purify_building:
        # Purification Building: 120 inert -> 30 eff
        f["purify"] = Formula(
            consumption=np.array([0, 0, 0, 0, 0, 0, -30, 120], dtype=float), output=0
        )
    if config.purify_node:
        # Test Area Purification Node: 30 sew -> 1 eff (max 12 multiples)
        f["purify_node"] = Formula(
            consumption=np.array([0, 0, 0, 0, 0, 30, -1, 0], dtype=float),
            output=0,
            limit=12,
        )

    for name, limit in config.formula_limits.items():
        if name in f:
            f[name].limit = limit
    for name, output in config.formula_outputs.items():
        if name in f:
            f[name].output = output

    return f


@dataclass
class SearchResult:
    """Result of search(). `all_candidates` is kept so callers can find
    near-optimal discrete (z, metatransfer) branches, e.g. to detect
    discrete-search ties that find_alternatives (an LP-objective-only tool)
    cannot see on its own."""

    result: OptimizeResult
    z: int
    metatransfer: np.ndarray
    formula_names: list[str]
    all_candidates: list[tuple[OptimizeResult, int, np.ndarray]]


def search(config: WulingConfig) -> SearchResult:
    """Search over forge allocations (z) and metatransfer choices, returning
    the best-dollar solution.

    Generalizes tests/wuling/test_wuling_1p2e.py::_search_1p2e.
    """
    formulas = build_formulas(config)
    candidates: list[tuple[OptimizeResult, int, np.ndarray]] = []
    for z in range(config.max_forges + 1):
        if not config.fix_hx_limit:
            formulas["hx"].limit = config.max_forges - z
        for mt in config.metatransfers:
            income = config.base_supply + z * XI_PER_FORGE + mt
            result = maximize_dollar(income, list(formulas.values()))
            candidates.append((result, z, mt.copy()))

    best_result, best_z, best_mt = max(candidates, key=lambda c: c[0].dollar_output)
    return SearchResult(
        result=best_result,
        z=best_z,
        metatransfer=best_mt,
        formula_names=list(formulas.keys()),
        all_candidates=candidates,
    )


def preset_1p2e_full() -> WulingConfig:
    """1.2e full: everything on (the default WulingConfig)."""
    return WulingConfig()


def preset_1p2e_equiv_1p2d() -> WulingConfig:
    """1.2e model with purify_node off, matching 1.2d's base/max_forges."""
    return WulingConfig(purify_node=False)


def preset_1p2_full() -> WulingConfig:
    """1.2e model reproducing the older "1.2 full" base/max_forges."""
    return WulingConfig(
        base_supply=(0, 480, 90, 180, 0, 0, 0, 0),
        max_forges=8,
        purify_node=False,
    )
