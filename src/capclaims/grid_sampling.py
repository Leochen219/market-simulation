"""Deterministic top-3, 5% support decoder for the Jev grid v2 probe."""
from __future__ import annotations
import hashlib, json, math
from collections.abc import Mapping, Sequence
from typing import Any
from capclaims.backends.jev import normalize_jev_probabilities

CUTOFF=0.05
TOLERANCE=1e-12
TOP_K=3
SCHEMA='top3_cutoff05_v1'


def decode(raw_probabilities:Mapping[str,Any], candidate_ids:Sequence[str], *,
           seed:str|int, person_id:str, repeat:int, stage:str)->dict[str,Any]:
    """Validate full raw Choice, retain top-3 candidates at raw p>=5%, draw once.

    All candidates remain in candidate_ids and the provider request. Cutoff is
    applied to raw probabilities with the explicitly frozen 1e-12 boundary
    tolerance. The draw is keyed only by the frozen semantic coordinates.
    """
    if not isinstance(raw_probabilities,Mapping): raise ValueError('probabilities must be a mapping')
    ids=list(candidate_ids)
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError('candidate IDs must be unique nonempty strings')
    # Existing Jev validation enforces complete keys and valid probability range.
    all_normalized,normalization=normalize_jev_probabilities(raw_probabilities,ids)
    if not isinstance(person_id,str) or not person_id or type(repeat)is not int or repeat not in (0,1):
        raise ValueError('invalid deterministic sampling coordinates')
    if not isinstance(stage,str) or not stage: raise ValueError('stage is required')
    if not isinstance(seed,(str,int)) or isinstance(seed,bool): raise ValueError('seed must be string or integer')
    raw={key:float(raw_probabilities[key]) for key in ids}
    eligible=[key for key in ids if raw[key]+TOLERANCE>=CUTOFF]
    ranked=sorted(eligible,key=lambda key:(-raw[key],ids.index(key)))[:TOP_K]
    retained_set=set(ranked)
    retained_mass=math.fsum(raw[key] for key in ranked)
    if not ranked or retained_mass<=0:
        raise ValueError('top3_cutoff05 has no surviving positive-mass candidate; no fallback is allowed')
    kept={key:(raw[key]/retained_mass if key in retained_set else 0.0) for key in ids}
    payload=json.dumps([SCHEMA,seed,person_id,repeat,stage],ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    digest=hashlib.sha256(payload).hexdigest()
    draw=(int(digest[:16],16)>>11)/2**53
    cumulative=0.0; selected=None
    for key in ids:
        probability=kept[key]
        if probability<=0: continue
        cumulative+=probability
        if draw<cumulative:
            selected=key; break
    if selected is None:
        # Roundoff at the final ULP only; this cannot add a candidate.
        selected=next(key for key in reversed(ids) if kept[key]>0)
    return {'schema':SCHEMA,'cutoff':CUTOFF,'tolerance':TOLERANCE,'top_k':TOP_K,
        'candidate_ids':ids,'raw_probabilities':raw,'all_candidate_normalized_probabilities':all_normalized,
        'normalization':normalization,'eligible_ids':eligible,'retained_ids':ranked,
        'cutoff_dropped_ids':[key for key in ids if key not in eligible],
        'topk_dropped_ids':[key for key in eligible if key not in retained_set],
        'cutoff_dropped_mass':math.fsum(raw[key] for key in ids if key not in eligible),
        'topk_dropped_mass':math.fsum(raw[key] for key in eligible if key not in retained_set),
        'retained_mass':retained_mass,'retained_normalized_probabilities':kept,
        'draw':draw,'draw_sha256':digest,'selected_id':selected}
