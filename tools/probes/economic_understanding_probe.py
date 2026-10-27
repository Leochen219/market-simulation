"""Predeclared actual-model numeric/economic readiness and history diagnostics.

No model answers are substituted. Oracle results are stored outside observations.
All economic choices have certain execution and no later trading/carryover.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from capclaims.backends.laya import LayaBackend
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, chosen_candidate, validate_reply, digest_json
from capclaims.paper.observations import build_paper_observation
from capclaims.randomness import canonical_json


def numeric_targets():
    sets = {
        "purchase": [0,1,37,137,173,200,299,332,333,334,400],
        "quantity": [1,12,93,246,444,555,666,777,888,999],
        "price": [1,10,11,71,111,222,333,334,444,555,666,777,888,999],
    }
    return [dict(field=field,target=target,confirmation=target in (37,246,334,888))
            for field, targets in sets.items() for target in targets]


def economic_cases():
    stop = dict(key="stop",side="STOP",qty=0,price=0)
    def case(name,p,D,Q,choices,expected,confirmation=False):
        return dict(id=name,p=p,D=D,Q=Q,w=10,confirmation=confirmation,
                    actions=[stop,*[dict(key=k,side=s,qty=q,price=m) for k,s,q,m in choices]],expected=expected)
    return [
        case("no_need_overpay",80,20,333,[("buy","buy",10,333)],"stop"),
        case("fill_shortage",80,20,10,[("buy","buy",10,20),("sell","sell",5,20)],"buy"),
        case("shortage_overpay",60,20,10,[("buy","buy",10,90)],"stop"),
        case("sell_surplus",70,20,60,[("sell","sell",20,15)],"sell"),
        case("sell_needed_premium",60,20,20,[("sell","sell",10,90),("buy","buy",10,90)],"sell"),
        case("keep_needed_below_value",80,20,20,[("sell","sell",10,20)],"stop"),
        case("expiry_zero_demand",90,0,10,[("buy","buy",1,1),("sell","sell",10,1)],"sell"),
        case("avoid_excess_buy",70,100,0,[("buy","buy",100,10),("excess","buy",300,10)],"buy"),
        case("confirmation_shortage",73,31,12,[("buy","buy",19,27),("sell","sell",12,27)],"buy",True),
        case("confirmation_surplus",61,8,40,[("sell","sell",32,17),("buy","buy",32,17)],"sell",True),
    ]


def action_profit_dollars(case, action):
    delta = action["qty"] * (1 if action["side"] == "buy" else -1 if action["side"] == "sell" else 0)
    return -case["w"]*case["Q"] - action["price"]*delta + case["p"]*min(case["D"],case["Q"]+delta)


def evaluate_readiness(numeric, economic):
    return bool(numeric and economic) and all(row.get("correct") is True for row in [*numeric,*economic])


def history(price):
    return [dict(round=r,p=70,Q=100,D=100,orders=[],profit_cents=600000,
                 net_quantity=0,mcp_cents=price*100) for r in (1,2)]


def observation(interface,field, *, p=80,D=20,Q=40,past=10):
    common = dict(agent_id="probe",round_index=2,treatment="GA",n_agents=10,
                  wholesale_price=10,p=p,history=history(past),partial_orders=(),
                  phase="procurement" if field == "purchase" else "orders",decision_field=field,
                  q=Q,d=D,slot=None if field == "purchase" else 0)
    if interface != "legacy_v1":
        common["decision_interface"] = interface
    return build_paper_observation(**common)


def permuted(request):
    return replace(request,candidates=tuple(Candidate(f"alt{i+7}",c.value,c.label)
                                            for i,c in enumerate(reversed(request.candidates))))


def numeric_requests(interface):
    from capclaims.paper.codec import prefix_mask
    from capclaims.paper.interface import build_candidates, numeric_observation
    pairs=[]
    for index, item in enumerate(numeric_targets()):
        field,target=item["field"],item["target"]
        for variant in ("original","reverse_remap"):
            for position,suffix in enumerate(("h","t","u")):
                prefix=f"{target:03d}"[:position]
                lo,hi=(0,400) if field == "purchase" else (1,999)
                obs=observation(interface,field)
                # Explicit target only tests representation, not strategy quality.
                obs["diagnostic_task"]=(f"Numeric representation check: represent the target {field} {target} exactly. "
                    "Choose the next digit whose final-number interval contains that target. Do not optimize a different value in this check.")
                obs=numeric_observation(obs,prefix,lo,hi,suffix)
                values=prefix_mask(prefix,lo,hi)
                if interface == "legacy_v1":
                    obs["legal_prefix_mask"] = list(values)
                request=DecisionRequest(f"numeric_{index:02d}_{variant}",0,f"paper:gate:{field}:{suffix}",obs,build_candidates(values,obs))
                if variant != "original": request=permuted(request)
                pairs.append((request,dict(family="numeric",field=field,target=target,suffix=suffix,
                                          variant=variant,confirmation=item["confirmation"],expected=int(f"{target:03d}"[position]))))
    return pairs


def economic_requests(interface):
    pairs=[]
    for case in economic_cases():
        for past in (10,333,777):
            for variant in ("original","reverse_remap"):
                obs=observation(interface,"side",p=case["p"],D=case["D"],Q=case["Q"],past=past)
                obs["diagnostic_task"]=("One-step economic check. Choose the action giving the highest guaranteed final-round profit. "
                    "Each quoted trade executes completely at its stated certain dollar price. Only this one action occurs; "
                    "there is no other current or future trading, no inventory carryover and no penalty for STOP. Past MCP cannot change these stated prices.")
                candidates=tuple(Candidate(f"c{i}",a["key"],"STOP: no trade" if a["side"] == "STOP"
                    else f'{a["side"].upper()} {a["qty"]} units at ${a["price"]} per unit') for i,a in enumerate(case["actions"]))
                request=DecisionRequest(f'{case["id"]}_{past}_{variant}',0,"paper:gate:economics",obs,candidates)
                if variant != "original": request=permuted(request)
                pairs.append((request,dict(family="economic",case=case["id"],past= past,variant=variant,
                                          confirmation=case["confirmation"],expected=case["expected"])))
    return pairs


async def execute_pairs(backend,pairs,output,tag):
    rows=[]
    # Groups of ten keep evidence bounded while still demonstrating real batches.
    for start in range(0,len(pairs),10):
        batch=pairs[start:start+10]
        replies=await asyncio.wait_for(backend.decide_many([r for r,_ in batch]),120)
        byid={r.request_id:r for r in replies}
        if len(byid)!=len(batch): raise ValueError("probe reply identity coverage mismatch")
        for request,meta in batch:
            reply=byid[request.request_id]
            validate_reply(request,reply)
            selected=chosen_candidate(request,reply,20261023,0)
            ordered=sorted(reply.probabilities.values(),reverse=True)
            strict=len(ordered)==1 or ordered[0]>ordered[1]
            row={**meta,"selected":selected.value,"correct":selected.value==meta["expected"] and strict,
                 "strict_ranking":strict,"request":asdict(request),"reply":asdict(reply)}
            rows.append(row)
        (output/f"{tag}-raw.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    return rows


async def history_sensitivity(backend, interface, output):
    """Matched uncertain production decisions; no invariance requirement or key."""
    from capclaims.paper.codec import prefix_mask
    from capclaims.paper.interface import build_candidates, numeric_observation
    rows=[]
    for field in ("purchase","price"):
        for past in (10,100,222,333,334,777):
            prefixes=[""]*10
            for suffix in ("h","t","u"):
                requests=[]
                for index,prefix in enumerate(prefixes):
                    obs=observation(interface,field,p=76,D=100,Q=40,past=past)
                    if field=="price":
                        obs["own_side"]="buy"
                        obs["own_qty_units" if interface=="economic_v2" else "own_qty"]=20
                    low,high=(0,400) if field=="purchase" else (1,999)
                    obs=numeric_observation(obs,prefix,low,high,suffix)
                    values=prefix_mask(prefix,low,high)
                    if interface == "legacy_v1":
                        obs["legal_prefix_mask"] = list(values)
                    requests.append(DecisionRequest(f"agent_{index:02d}",2,f"paper:sensitivity:{field}:{suffix}",obs,build_candidates(values,obs)))
                replies=await asyncio.wait_for(backend.decide_many(requests),120)
                if len(replies)!=10: raise ValueError("sensitivity requires ten complete replies")
                byid={r.request_id:r for r in replies}
                for index,request in enumerate(requests):
                    reply=byid[request.request_id]
                    validate_reply(request,reply)
                    chosen=chosen_candidate(request,reply,20261023,0)
                    prefixes[index]+=str(chosen.value)
                    rows.append(dict(field=field,past=past,request=asdict(request),reply=asdict(reply),selected=chosen.value))
            rows.append(dict(field=field,past=past,completed_values=[int(p) for p in prefixes]))
            (output/f"{interface}-sensitivity-raw.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    return [r for r in rows if "completed_values" in r]


async def production_capability_stage(backend, interface, output, suffix):
    from capclaims.paper.interface import build_candidates
    from tools.probes.paper_capability_probe import build_paper_states
    states=[json.loads(s) for s in build_paper_states(interface)]
    from capclaims.paper.interface import numeric_observation
    prefix={"h":"","t":"9","u":"99"}[suffix]
    states=[numeric_observation(s,prefix,1,999,suffix) for s in states]
    requests=[DecisionRequest(f"agent_{i:02d}",18,f"paper:slot4:price:{suffix}",s,build_candidates(tuple(range(10)),s))
              for i,s in enumerate(states)]
    batch=await asyncio.wait_for(backend.decide_many(requests),120)
    singles=[(await asyncio.wait_for(backend.decide_many([r]),120))[0] for r in requests]
    changed=list(requests)
    changed[0]=replace(changed[0],observation=dict(changed[0].observation,own_p=51))
    perturbed=await asyncio.wait_for(backend.decide_many(changed),120)
    for request,reply in zip(requests,batch,strict=True): validate_reply(request,reply)
    parity=max(abs(b.probabilities[k]-s.probabilities[k]) for b,s in zip(batch,singles) for k in b.probabilities)
    isolation=max(abs(batch[i].probabilities[k]-perturbed[i].probabilities[k]) for i in range(1,10) for k in batch[i].probabilities)
    shapes=sorted({size for b in batch for size in b.timing["forward_batch_sizes"]})
    untruncated=all(not b.usage["truncated"] and b.usage["state_tokens_dropped"]==0 for b in [*batch,*singles,*perturbed])
    result=dict(status="PASS" if shapes==[10] and parity<=1e-4 and isolation<=1e-4 and untruncated else "FAIL",
                forward_batch_sizes=shapes,batch_single_max_error=parity,other_row_isolation_error=isolation,
                max_input_tokens=max(b.usage["input_tokens"] for b in batch),
                max_state_tokens=max(b.usage["state_tokens"] for b in batch),untruncated=untruncated)
    (output/f"{interface}-production-capability-{suffix}.json").write_text(json.dumps(dict(result=result,batch=[asdict(b) for b in batch],singles=[asdict(b) for b in singles],perturbed=[asdict(b) for b in perturbed]),indent=2)+"\n",encoding="utf-8")
    return result


async def production_capability(backend,interface,output):
    stages={}
    for suffix in ("h","t","u"):
        stages[suffix]=await production_capability_stage(backend,interface,output,suffix)
    return dict(status="PASS" if all(s["status"]=="PASS" for s in stages.values()) else "FAIL",stages=stages)


async def main_async(args):
    output=args.output_dir
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    fixtures={"numeric":numeric_targets(),"economic":economic_cases(),"history_prices":[10,333,777],
              "acceptance":"Every numeric and strict guaranteed-profit answer correct, including confirmations and ID/order controls."}
    (output/"frozen-fixtures.json").write_text(json.dumps(fixtures,indent=2)+"\n",encoding="utf-8")
    result={"status":"FAIL","suite":args.suite,"fixtures_hash":digest_json(fixtures),"probe_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"interfaces":{}}
    backend=LayaBackend(args.model_path,args.device)
    try:
        for interface in args.interfaces.split(","):
            capability=await production_capability(backend,interface,output)
            def selected(pairs):
                return [pair for pair in pairs if pair[1]["confirmation"] == (args.suite == "confirmation")]
            numeric=await execute_pairs(backend,selected(numeric_requests(interface)),output,interface+"-numeric")
            econ=await execute_pairs(backend,selected(economic_requests(interface)),output,interface+"-economic")
            summary={"status":"PASS" if evaluate_readiness(numeric,econ) and capability["status"]=="PASS" else "FAIL",
                "production_capability":capability,
                "numeric_correct":sum(r["correct"] for r in numeric),"numeric_total":len(numeric),
                "economic_correct":sum(r["correct"] for r in econ),"economic_total":len(econ),
                "confirmation_numeric_correct":sum(r["correct"] for r in numeric if r["confirmation"]),
                "confirmation_numeric_total":sum(r["confirmation"] for r in numeric),
                "confirmation_economic_correct":sum(r["correct"] for r in econ if r["confirmation"]),
                "confirmation_economic_total":sum(r["confirmation"] for r in econ),
                "failures":[{k:r[k] for k in ("family","selected","expected","variant","confirmation")} | {k:r[k] for k in ("case","past","field","target","suffix") if k in r} for r in [*numeric,*econ] if not r["correct"]]}
            if args.suite=="development":
                summary["history_sensitivity"]=await history_sensitivity(backend,interface,output)
            result["interfaces"][interface]=summary
            (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        result["status"]="PASS" if result["interfaces"] and all(s["status"]=="PASS" for s in result["interfaces"].values()) else "FAIL"
    except Exception as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        await backend.aclose()
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="PASS" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",default=".cache/models/laya-multilingual")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--interfaces",default="legacy_v1,economic_v2")
    parser.add_argument("--suite",choices=("development","confirmation"),default="development")
    parser.add_argument("--output-dir",type=Path,required=True)
    return asyncio.run(main_async(parser.parse_args()))


if __name__=="__main__":
    raise SystemExit(main())
