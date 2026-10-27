from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.paper.settlement import settle_paper_round
from capclaims.paper.observations import build_paper_observation
from capclaims.paper.auction import clear_paper_auction


def test_nv_procurement_explicitly_tells_agent_there_is_no_future_market():
    common = dict(agent_id="a", round_index=0,n_agents=10,wholesale_price=10,p=80,
                  history=(),partial_orders=(),phase="procurement")
    nv = build_paper_observation(**common,treatment="NV")
    ga = build_paper_observation(**common,treatment="GA")
    assert nv != ga
    assert nv["maximum_total_orders"] == 0
    assert "no secondary trading" in nv["public_instructions"]["trading"]
    assert "own_d" not in nv


def test_observation_hides_during_procurement_and_shows_projection_after_demand():
    common = dict(agent_id="a", round_index=0, treatment="GA", n_agents=10,
                  wholesale_price=10, p=80, history=(), partial_orders=())
    procurement = build_paper_observation(**common, phase="procurement")
    assert "own_d" not in procurement
    post = build_paper_observation(**common, phase="orders", q=137, d=100)
    assert post["pretrade_profit_cents"] == -100 * 10 * 137 + 100 * 80 * 100
    assert post["own_q"] == 137 and post["own_d"] == 100
    assert "other_agents" not in post and "group_members" not in post
    compact = build_paper_observation(**dict(common, partial_orders=(
        {"order_id": "r000:a:slot0", "owner": "a", "side": "sell", "qty": 20, "limit_dollars": 325},)),
        phase="orders", q=137, d=100)
    assert compact["partial_orders"] == [{"slot": 0, "side": "sell", "qty": 20, "limit_dollars": 325}]
    assert "owner" not in compact["partial_orders"][0] and "order_id" not in compact["partial_orders"][0]
    assert "midpoint of the last matched bid and ask" in compact["public_instructions"]["clearing"]


def test_settlement_uses_net_market_flow_and_nonbinding_cash_bound():
    orders = [PaperOrder("b", "buyer", "buy", 3, 999), PaperOrder("s", "seller", "sell", 3, 1)]
    auction = clear_paper_auction(orders)
    result = settle_paper_round({"buyer": (80, 20, 10), "seller": (70, 0, 10)}, auction, 10)
    assert result.profits_cents["buyer"] == -10000 - auction.price_cents * 3 + 80 * 13 * 100
    assert result.profits_cents["seller"] == -10000 + auction.price_cents * 3
    assert result.issued_units == result.redeemed_units + result.burned_units
    assert result.cash_grant_cents >= 501000500


def test_gross_sells_are_validated_before_matching_and_buys_do_not_fund_them():
    portfolio = [PaperOrder("buy", "a", "buy", 999, 999), PaperOrder("s1", "a", "sell", 2, 1)]
    try:
        validate_portfolio(portfolio, "a", 1)
    except ValueError as exc:
        assert "gross sell" in str(exc)
    else:
        raise AssertionError("buy order incorrectly increased same-round sell cap")
