import asyncio,json
from copy import deepcopy
import pytest
from capclaims.sell_oracle import design,source,runner_v2 as runner

def context(i=0):
 return {'person_id':f'p{i}','formal_round':11,'current_state':{'round':11,'p':80,'w':10,'Q':100,'D':20},'history':[{'round':n,'p':70,'Q':100,'D':30,'orders':[{'side':'sell','qty':6,'price':12}],'profit_cents':100} for n in range(1,11)]}

def make_plan(n=1,duplicate=False):
 contexts=[context(i) for i in range(n)];components=[]
 for c in contexts:components.extend(design.components(c,[6,6,14] if duplicate else [6]))
 queries=design.queries(components);return {'contexts':contexts,'components':components,'queries':queries,'schedule':queries}

def test_exact_quantity_dedup_ids_and_label_isolation():
 async def exercise():
  c=context();before=deepcopy(c); components=design.components(c,[6,14,6]); queries=design.queries(components)
  assert len(queries)==2 and components[0]['query_id']==components[2]['query_id'] and components[0]['component_id']!=components[2]['component_id']
  assert components==design.components(c,[6,14,6]); assert c==before
  a=await runner.body(c,queries[0]); state=design.state(c,queries[0]['quantity'])
  assert state['supplied_quantity_units'] in (6,14) and not {'slot','target','market_size','selected_order_count','partial_orders','model_chosen_quantity'}&set(state)
  assert len(design.OPTIONS)==199 and list(design.OPTIONS)[0]=='p005' and list(design.OPTIONS)[-1]=='p995'
  assert b'component_id' not in a and b'query_id' not in a and b'occurrence' not in a
  labels=[12,17,900]; altered=[999,1,5]; assert labels!=altered
  assert design.components(c,[6,14,6])==components and await runner.body(c,queries[0])==a
  raw={k:1/199 for k in design.OPTIONS};assert design.decide(raw,components[0])['status']=='client_numeric_abstain'
  with pytest.raises(ValueError):design.components(c,[101])
 asyncio.run(exercise())

def test_ledger_106_cap(tmp_path):
 ledger=runner.Ledger(tmp_path/'ledger.sqlite',preview_hash='a'*64)
 for i in range(106):ledger.reserve(str(i))
 assert ledger.summary()['committed_nanodollars']==106*runner.RESERVATION<runner.BUDGET
 with pytest.raises(ValueError):ledger.reserve('extra')

def test_10_distinct_concurrent_calls(tmp_path):
 class Client:
  halted=False
  def __init__(self):self.n=0;self.active=0;self.peak=0;self.ready=asyncio.Event();self.ids=set();self.body_hashes=set()
  async def send_body(self,*,request_id,body):
   self.ids.add(request_id);self.body_hashes.add(runner.sha(body));self.n+=1;self.active+=1;self.peak=max(self.peak,self.active)
   if self.n==10:self.ready.set()
   await self.ready.wait();self.active-=1
   return {'answers':{'decision':{'choice':'p005','probabilities':{k:1/199 for k in design.OPTIONS}}}}
 async def exercise():
  plan=make_plan(10);client=Client()
  for i,c in enumerate(plan['contexts']): c['current_state']['p']=70+i
  ctx={(c['person_id'],c['formal_round']):c for c in plan['contexts']}
  rows=await asyncio.gather(*(runner.call(client,tmp_path,ctx[(q['person_id'],q['formal_round'])],q,[c for c in plan['components'] if c['query_id']==q['query_id']]) for q in plan['queries']))
  assert len(rows)==client.peak==len(client.ids)==len(client.body_hashes)==10 and all(next(iter(r['decisions'].values()))['status']=='client_numeric_abstain' for r in rows)
 asyncio.run(exercise())

def test_missing_key_no_marker(tmp_path,monkeypatch):
 import dotenv
 async def valid(_):return {}
 monkeypatch.setattr(runner,'valid_plan',valid);monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:None);monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
 with pytest.raises(ValueError,match='provider key missing'):asyncio.run(runner.run(tmp_path))
 assert not (tmp_path/'started').exists() and not (tmp_path/'cost-ledger.sqlite').exists()

def test_mock_complete_dedup_replay_and_tamper(tmp_path,monkeypatch):
 import httpx2
 async def exercise():
  plan=make_plan(10,True);ctx={(c['person_id'],c['formal_round']):c for c in plan['contexts']};bodies={}
  for q in plan['queries']:
   data=await runner.body(ctx[(q['person_id'],q['formal_round'])],q);bodies[q['query_id']]={'sha256':runner.sha(data),'bytes':len(data)}
  plan['bodies']=bodies
  async def valid(_):return plan
  monkeypatch.setattr(runner,'valid_plan',valid);(tmp_path/'plan.json').write_bytes(b'plan'); h=runner.sha(b'plan');(tmp_path/'started').write_text(h+'\n',encoding='ascii');ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
  async def reply(req):
   body=json.loads(await req.aread());ids=body['questions']['decision']['criteria'];first=next(iter(ids))
   return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':first,'probabilities':{k:float(k==first) for k in ids}}}})
  async with runner.JevWireClient(api_key='dummy',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(reply)) as client:
   async def no_rate():pass
   client._rate_slot=no_rate
   for i in range(0,len(plan['queries']),10):
    await asyncio.gather(*(runner.call(client,tmp_path,ctx[(q['person_id'],q['formal_round'])],q,[c for c in plan['components'] if c['query_id']==q['query_id']]) for q in plan['queries'][i:i+10]))
  runner.append(tmp_path/'status.jsonl',{'status':'COMPLETE','physical_requests':len(plan['queries'])})
  _,preds,status=await runner.replay(tmp_path);assert len(preds)==30 and len(plan['queries'])==20 and all(p['price']==5 for p in preds)
  folder=tmp_path/'wire-evidence'/runner.sha(plan['queries'][0]['query_id'].encode());(folder/'response.json').write_bytes(b'{}')
  with pytest.raises(ValueError):await runner.replay(tmp_path)
 asyncio.run(exercise())

def test_http_failure_retains_reservation(tmp_path,monkeypatch):
 import httpx2
 async def exercise():
  plan=make_plan(); q=plan['queries'][0];c=plan['contexts'][0];data=await runner.body(c,q);plan['bodies']={q['query_id']:{'sha256':runner.sha(data),'bytes':len(data)}}
  async def valid(_):return plan
  monkeypatch.setattr(runner,'valid_plan',valid);(tmp_path/'plan.json').write_bytes(b'plan');h=runner.sha(b'plan');(tmp_path/'started').write_text(h+'\n',encoding='ascii');ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
  async def fail(req):return httpx2.Response(500,content=b'failure')
  async with runner.JevWireClient(api_key='dummy',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(fail)) as client:
   with pytest.raises(ValueError):await runner.call(client,tmp_path,c,q,plan['components'])
  runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':1});_,preds,status=await runner.replay(tmp_path)
  assert status['status']=='PARTIAL_TECHNICAL' and not preds and ledger.summary()['reserved_nanodollars']==runner.RESERVATION
  folder=tmp_path/'wire-evidence'/runner.sha(q['query_id'].encode());(folder/'request.json').write_bytes(b'bad')
  with pytest.raises(ValueError):await runner.replay(tmp_path)
 asyncio.run(exercise())


def test_target_price_mutation_changes_only_labels_and_later_history():
 async def exercise():
  c=context();orders=[{'side':'sell','qty':6,'price':5},{'side':'sell','qty':6,'price':40},{'side':'sell','qty':14,'price':9}]
  changed=deepcopy(orders);changed[0]['price']=999;changed[1]['price']=1
  a,labels=source.split_components(c,orders);b,other=source.split_components(c,changed)
  assert a==b and labels!=other and design.queries(a)==design.queries(b)
  assert [await runner.body(c,q) for q in design.queries(a)]==[await runner.body(c,q) for q in design.queries(b)]
  raw={k:0.0 for k in design.OPTIONS};raw['p005']=raw['p010']=0.5
  assert [design.decide(raw,x) for x in a]==[design.decide(raw,x) for x in b]
  later=context();later['formal_round']=12;later['current_state']['round']=12;later['history']=later['history'][1:]+[{'round':11,'p':80,'Q':100,'D':20,'orders':orders,'profit_cents':100}]
  later_changed=deepcopy(later);later_changed['history'][-1]['orders']=changed
  assert design.state(later,6)!=design.state(later_changed,6)
 asyncio.run(exercise())

def test_predispatch_construction_and_marker_replay(tmp_path,monkeypatch):
 async def exercise():
  plan=make_plan();q=plan['queries'][0];c=plan['contexts'][0];data=await runner.body(c,q);plan['bodies']={q['query_id']:{'sha256':runner.sha(data),'bytes':len(data)}}
  async def valid(_):return plan
  monkeypatch.setattr(runner,'valid_plan',valid);(tmp_path/'plan.json').write_bytes(b'plan');h=runner.sha(b'plan');(tmp_path/'started').write_text(h+'\n',encoding='ascii');runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
  original=runner.body
  async def broken(*args):raise ValueError('construction')
  monkeypatch.setattr(runner,'body',broken)
  class Client:halted=False
  client=Client()
  with pytest.raises(ValueError):await runner.call(client,tmp_path,c,q,plan['components'])
  assert client.halted
  runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':0,'phase':'dispatch'});monkeypatch.setattr(runner,'body',original)
  _,preds,status=await runner.replay(tmp_path);assert not preds and status['status']=='PARTIAL_TECHNICAL'
  with pytest.raises(FileExistsError):await runner.run(tmp_path)
 asyncio.run(exercise())

def test_invalid_probability_is_technical_failure_after_settlement(tmp_path,monkeypatch):
 import httpx2
 async def exercise():
  plan=make_plan();q=plan['queries'][0];c=plan['contexts'][0];data=await runner.body(c,q);plan['bodies']={q['query_id']:{'sha256':runner.sha(data),'bytes':len(data)}}
  async def valid(_):return plan
  monkeypatch.setattr(runner,'valid_plan',valid);(tmp_path/'plan.json').write_bytes(b'plan');h=runner.sha(b'plan');(tmp_path/'started').write_text(h+'\n',encoding='ascii');ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
  async def reply(req):return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':'p005','probabilities':{'p005':1}}}})
  async with runner.JevWireClient(api_key='dummy',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(reply)) as client:
   with pytest.raises(ValueError):await runner.call(client,tmp_path,c,q,plan['components'])
  runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':1});_,preds,status=await runner.replay(tmp_path)
  assert not preds and status['status']=='PARTIAL_TECHNICAL' and ledger.summary()['settled_nanodollars']==420
 asyncio.run(exercise())

def test_phase_guards_reject_duplicates_and_state_mismatch():
 from types import SimpleNamespace
 p=SimpleNamespace(scope='GA 10',treatment='GA',participant_id='p',round=1,phase='procurement',observed={'p':80,'w':10},action={'Q':100})
 o=SimpleNamespace(scope='GA 10',treatment='GA',participant_id='p',round=1,phase='orders',observed={'p':80,'w':10,'Q':100},action={'orders':[]})
 assert len(source.validate_phases([p,o]))==1
 with pytest.raises(ValueError,match='duplicate'):source.validate_phases([p,o,o])
 with pytest.raises(ValueError,match='missing'):source.validate_phases([p])
 wrong=deepcopy(o);wrong.observed['Q']=101
 with pytest.raises(ValueError,match='state mismatch'):source.validate_phases([p,wrong])
 wrong=deepcopy(o);wrong.observed['w']=50
 with pytest.raises(ValueError,match='state mismatch'):source.validate_phases([p,wrong])

def test_history_money_uses_exact_cents():
 c=context();c['history'][0]['profit_cents']=1234567890123456789
 assert design.state(c,6)['history'][0]['profit_dollars']=='12345678901234567.89'

def test_v1_key_order_bug_v2_stable_and_candidate_order():
 from capclaims.sell_oracle import runner as v1
 async def exercise():
  c=context();q=design.queries(design.components(c,[6]))[0];reloaded=json.loads(runner.canonical(c))
  a=await v1.body(c,q);b=await v1.body(reloaded,q)
  assert json.loads(a)==json.loads(b) and a!=b
  v2a=await runner.body(c,q);v2b=await runner.body(reloaded,q)
  assert v2a==v2b and list(json.loads(v2a)['questions']['decision']['criteria'])==list(design.OPTIONS)
 asyncio.run(exercise())

def test_actual_freeze_reload_mock_run_replay_all_bodies(tmp_path,monkeypatch):
 import dotenv,httpx2
 original=runner.JevWireClient
 async def exercise():
  cs=[context(i) for i in range(24)]
  for i,c in enumerate(cs):
   c['history'][0]['orders']=[{'side':'sell','qty':6+i%8,'price':10+i}]
   c['current_state']['p']=70+i
  comps=[x for c in cs for x in design.components(c,[6])];labels={x['component_id']:17 for x in comps};roster=[c['person_id'] for c in cs]
  monkeypatch.setattr(source,'load',lambda:(roster,cs,comps,labels,{'synthetic':True}))
  root=tmp_path/'new-study';await runner.freeze(root);plan=await runner.valid_plan(root)
  ix={(c['person_id'],c['formal_round']):c for c in plan['contexts']}
  async def construct(q):return await runner.body(ix[(q['person_id'],q['formal_round'])],q)
  seq=[await construct(q) for q in plan['schedule']]; concurrent=[]
  for i in range(0,len(plan['schedule']),10):concurrent.extend(await asyncio.gather(*(construct(q) for q in plan['schedule'][i:i+10])))
  assert seq==concurrent
  assert all({'sha256':runner.sha(b),'bytes':len(b)}==plan['bodies'][q['query_id']] for q,b in zip(plan['schedule'],seq))
  inflight=0;peak=0;first_hashes=set();first_ready=asyncio.Event()
  async def reply(req):
   nonlocal inflight,peak
   payload=await req.aread();inflight+=1;peak=max(peak,inflight)
   if len(first_hashes)<10:
    first_hashes.add(runner.sha(payload))
    if len(first_hashes)==10:first_ready.set()
    await first_ready.wait()
   inflight-=1
   request=json.loads(payload);ids=request['questions']['decision']['criteria'];first=next(iter(ids));return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':first,'probabilities':{k:float(k==first) for k in ids}}}})
  class MockClient(original):
   def __init__(self,**kw):super().__init__(**kw,transport=httpx2.MockTransport(reply))
   async def _rate_slot(self):pass
  monkeypatch.setattr(runner,'JevWireClient',MockClient);monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**k:None);monkeypatch.setenv('TYPESAFE_API_KEY','offline-dummy')
  status=await runner.run(root);_,predictions,replay_status=await runner.replay(root)
  assert status['status']==replay_status['status']=='COMPLETE' and status['physical_requests']==24 and len(predictions)==24
  assert peak==10 and len(first_hashes)==10
 asyncio.run(exercise())

def test_preparation_mismatch_private_replay_and_tamper(tmp_path,monkeypatch):
 async def exercise():
  plan=make_plan();q=plan['queries'][0];c=plan['contexts'][0];actual=await runner.body(c,q);plan['bodies']={q['query_id']:{'sha256':'f'*64,'bytes':len(actual)}}
  async def valid(_):return plan
  monkeypatch.setattr(runner,'valid_plan',valid);(tmp_path/'plan.json').write_bytes(b'plan');h=runner.sha(b'plan');(tmp_path/'started').write_text(h+'\n',encoding='ascii');ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
  class Client:
   halted=False
   async def send_body(self,**kw):pytest.fail('must not send')
  with pytest.raises(ValueError):await runner.call(Client(),tmp_path,c,q,plan['components'],plan['bodies'][q['query_id']])
  assert ledger.summary()['physical_requests']==0
  runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':0,'phase':'dispatch'})
  # The intended body must still be independently reconstructible during replay.
  original=runner.body
  async def intended(*args): return b'different frozen request'
  expected=await intended();plan['bodies'][q['query_id']]={'sha256':runner.sha(expected),'bytes':len(expected)}
  monkeypatch.setattr(runner,'body',intended)
  _,pred,status=await runner.replay(tmp_path);assert not pred and status['status']=='PARTIAL_TECHNICAL'
  folder=tmp_path/'preparation-evidence'/ (runner.sha(q['query_id'].encode())+'.json');folder.write_bytes(b'tamper')
  with pytest.raises(ValueError):await runner.replay(tmp_path)
 asyncio.run(exercise())

def test_client_enter_failure_real_lifecycle_replays_partial(tmp_path,monkeypatch):
 import dotenv
 async def exercise():
  c=context();cs=design.components(c,[6]);labels={cs[0]['component_id']:17}
  monkeypatch.setattr(source,'load',lambda:([c['person_id']],[c],cs,labels,{'synthetic':True}))
  root=tmp_path/'client-start-study';await runner.freeze(root)
  class BrokenClient:
   def __init__(self,**kwargs):pass
   async def __aenter__(self):raise ValueError('offline simulated startup failure')
   async def __aexit__(self,*args):pass
  monkeypatch.setattr(runner,'JevWireClient',BrokenClient)
  monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**kw:None);monkeypatch.setenv('TYPESAFE_API_KEY','offline-dummy')
  with pytest.raises(ValueError,match='offline simulated startup failure'):await runner.run(root)
  _,predictions,status=await runner.replay(root)
  assert status['status']=='PARTIAL_TECHNICAL' and status['phase']=='client_start' and status['physical_requests']==0 and not predictions
  assert runner.Ledger(root/'cost-ledger.sqlite',preview_hash=runner.sha((root/'plan.json').read_bytes())).summary()['physical_requests']==0
 asyncio.run(exercise())
