"""Version 2 reports use the shared pure oracle analysis with the V2 replay."""
import asyncio,json
from pathlib import Path
from capclaims.sell_oracle.analysis import summarize,public_aggregate
from capclaims.flat_grid_study.analysis import accounting
from capclaims.sell_oracle import runner_v2 as runner

async def report_async(root,output):
    root=Path(root);plan,predictions,status=await runner.replay(root)
    if status['status']!='COMPLETE': raise ValueError('refuse complete-score report for technical-partial study')
    labels=json.loads((root/'labels-private.json').read_text(encoding='utf-8'))
    full=summarize(plan,predictions,labels)
    full.update(status=status,plan_sha256=runner.sha((root/'plan.json').read_bytes()),accounting=accounting(root))
    (root/'analysis-private.json').write_text(json.dumps(full,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(public_aggregate(full),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return full

def report(root=None,output=None):
    return asyncio.run(report_async(root or runner.OUT,output or runner.ROOT/'docs/analysis/jev-expanded-sell-oracle-v2-2026-10-28.json'))
