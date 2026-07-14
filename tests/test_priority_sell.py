from factorylib.priority_sell import allocate_by_priority


def test_full_budget_sells_everything():
    sold, unsold = allocate_by_priority({"a": 10.0, "b": 20.0}, ["a", "b"], 1000.0)
    assert sold == {"a": 10.0, "b": 20.0}
    assert unsold == {"a": 0.0, "b": 0.0}


def test_budget_exhausted_partway_through_priority_list():
    sold, unsold = allocate_by_priority({"a": 100.0, "b": 50.0}, ["a", "b"], 120.0)
    assert sold == {"a": 100.0, "b": 20.0}
    assert unsold == {"a": 0.0, "b": 30.0}


def test_zero_budget_sells_nothing():
    sold, unsold = allocate_by_priority({"a": 10.0}, ["a"], 0.0)
    assert sold == {"a": 0.0}
    assert unsold == {"a": 10.0}


def test_names_not_in_priority_list_come_last_in_original_order():
    sold, _ = allocate_by_priority({"a": 10.0, "b": 10.0, "c": 10.0}, ["c"], 25.0)
    # c is prioritized first (=10 sold, 15 remaining); a/b follow in dict
    # order, so a gets fully sold (5 remaining) and b gets whatever's left.
    assert sold["c"] == 10.0
    assert sold["a"] == 10.0
    assert sold["b"] == 5.0


def test_priority_names_absent_from_amounts_get_zero_entries():
    """A priority name with nothing to sell (0.0) still gets an entry in
    both dicts -- harmless, and lets callers iterate the full priority
    list without a KeyError."""
    sold, unsold = allocate_by_priority({"b": 10.0}, ["a", "b"], 100.0)
    assert sold == {"a": 0.0, "b": 10.0}
    assert unsold == {"a": 0.0, "b": 0.0}


def test_worked_example_matches_wuling_refined_solution():
    """The exact numbers from the real bug report: 1278 $/min produced
    across 6 goods, 1090 $/min stock-bill cap, sold in Yazhen A >
    Hetonite Part > Heavy Xiranite > SC Wuling Battery > Yazhen C >
    everything else order."""
    amounts = {
        "sc": 648.0,
        "hx": 324.0,
        "ya": 132.0,
        "cp_sell": 120.0,
        "xi_sell": 30.0,
        "yc": 24.0,
    }
    priority = ["ya", "hp", "hx", "sc", "yc"]
    sold, unsold = allocate_by_priority(amounts, priority, 1090.0)
    # hp isn't in `amounts` (it's not produced in this scenario), but
    # still gets a harmless zero entry -- see the priority-names-absent
    # test above.
    assert sold == {
        "hp": 0.0,
        "sc": 634.0,
        "hx": 324.0,
        "ya": 132.0,
        "cp_sell": 0.0,
        "xi_sell": 0.0,
        "yc": 0.0,
    }
    assert unsold == {
        "hp": 0.0,
        "sc": 14.0,
        "hx": 0.0,
        "ya": 0.0,
        "cp_sell": 120.0,
        "xi_sell": 30.0,
        "yc": 24.0,
    }
    assert sum(sold.values()) == 1090.0
    assert sum(unsold.values()) == 1278.0 - 1090.0
