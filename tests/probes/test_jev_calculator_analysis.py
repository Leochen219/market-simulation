from tools.analysis.analyze_jev_calculator import summarize_arm


def test_unresolved_choices_are_excluded_from_payoff_means_but_count_as_failed_unique_decisions():
    base = dict(economic_tie=False, tool_results=[], model_tool_calls=0,
                all_actions_calculated=False, payoffs_cents={'c0':100,'c1':200},
                calls=[{'request':{'candidates':[{'candidate_id':'c0','value':'STOP'},
                                               {'candidate_id':'c1','value':'SELL'}]}}])
    rows = [{**base, 'selected':'STOP','selected_id':'c0','correct':False,'regret_cents':100,'profit_cents':100},
            {**base, 'selected':'TOP_TIE','selected_id':None,'correct':False,'regret_cents':None,'profit_cents':None}]
    result = summarize_arm(rows)
    assert result['unique_correct'] == 0 and result['unique_n'] == 2
    assert result['resolved_unique_n'] == 1
    assert result['mean_regret_cents_resolved_unique'] == 100
    assert result['favorable_sell_n'] == 2 and result['favorable_sell_stop'] == 1
    assert result['probability_ties'] == 1
