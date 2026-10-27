from tools.probes.paper_capability_probe import build_paper_states, full_integer_criteria


def test_uniform_digit_scores_do_not_claim_high_margin_agreement():
    from tools.probes.paper_capability_probe import margin_agreement
    assert margin_agreement([{'a':.5,'b':.5}], [{'a':.5,'b':.5}]) == (0,None)
    assert margin_agreement([{'a':.8,'b':.2}], [{'a':.2,'b':.8}]) == (1,0.0)


def test_expanded_fixture_is_private_heterogeneous_and_covers_five_order_history():
    states = build_paper_states()
    assert len(states) == len(set(states)) == 10
    import json
    rows = [json.loads(state) for state in states]
    assert max(len(row['own_history']) for row in rows) == 2
    assert {len(row['own_history']) for row in rows} == {0, 1, 2}
    assert all(len(record['orders']) == 5 for row in rows for record in row['own_history'])
    assert all(len(row['partial_orders']) == 4 for row in rows)
    assert all('public_instructions' in row for row in rows)
    assert all('other_agents' not in row and 'world_hash' not in row for row in rows)


def test_full_integer_criteria_do_not_silently_keep_old_grids():
    purchase = full_integer_criteria(0, 400)
    price = full_integer_criteria(1, 999)
    assert len(purchase) == 401 and len(price) == 999
    assert purchase['v137'] == '137'
    assert price['v325'] == '325' and price['v999'] == '999'
