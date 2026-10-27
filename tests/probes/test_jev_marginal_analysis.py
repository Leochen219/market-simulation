from tools.analysis.analyze_jev_marginal_value import average_value_action


def test_exploratory_average_rule_differs_from_true_marginal_inventory_value():
    row = dict(Q=40, D=20, p=70, m=15)
    assert average_value_action(row) == "STOP"  # profitable surplus SELL is missed
    assert average_value_action(dict(row, D=0)) == "SELL"
    assert average_value_action(dict(row, m=35)) == "STOP"
    assert average_value_action(dict(row, m=35), weak=True) == "SELL"


def test_exploratory_rule_keeps_its_counterexample_in_needed_inventory_case():
    assert average_value_action(dict(Q=20, D=20, p=100, m=70)) == "STOP"
