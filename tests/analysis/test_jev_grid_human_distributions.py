import copy
import pytest
from tools.analysis.compare_jev_grid_human_distributions import weighted,w1,project,energy,bounds,common_slots

def test_weighted_w1_and_balanced_orders():
    human=[[0,10,20,30],[100]]; before=copy.deepcopy(human)
    h=weighted(human); m=weighted([[10],[100]])
    assert sum(w for x,w in h[:4])==.5
    assert w1(h,m)==5.0
    assert human==before and w1(h,m)==w1(h,list(reversed(m)))

def test_grid_ties_clamp_and_sell_capacity():
    assert project(15,3.75,'BUY',10)==(20,5)
    assert project(999,999,'BUY',10)==(990,997.5)
    assert project(999,1,'SELL',37)==(30,2.5)

def test_energy_and_missing_envelope():
    a=weighted([[(10,10)],[(20,20)]])
    assert energy(a,a)==pytest.approx(0)
    h=weighted([[0],[10]]); obs=[(0,.5)]
    lo,hi=bounds(h,obs,.5,0,10)
    assert lo==pytest.approx(0) and hi==pytest.approx(5)


def test_common_mask_intersects_all_three_arms():
    arms=[{('p',r,'price'):{'record_type':'STEP'} for r in (0,1)} for _ in range(3)]
    arms[1][('p',0,'price')]={'record_type':'DECODE_FAILURE'}
    assert common_slots(arms,{'p'})=={('p',1)}


def test_model_common_mask_keeps_equal_person_weight():
    full=weighted([[10,20],[30,40]])
    common=weighted([[10],[30,40]])
    assert full==[(10,.25),(20,.25),(30,.25),(40,.25)]
    assert common==[(10,.5),(30,.25),(40,.25)]
    assert sum(weight for _,weight in common)==1
