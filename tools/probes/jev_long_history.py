"""Offline preview, freeze, run, replay and report for the history study."""
import argparse,asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.long_history import runner,analysis,design

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('preview','freeze','run','replay','report'));p.add_argument('--plan-dir',type=Path,default=runner.OUT);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.command=='preview':
        c,t,m,size=asyncio.run(runner.preview());result={'contexts':len(c),'paths':len(design.schedule(c)),'max_body_bytes':size}
    elif a.command=='freeze':result=asyncio.run(runner.freeze(a.plan_dir))
    elif a.command=='run':result=asyncio.run(runner.run(a.plan_dir))
    elif a.command=='replay':
        _,pred,status=runner.replay(a.plan_dir);result={'status':status,'replayed_predictions':len(pred)}
    else:
        if a.output is None:p.error('--output required')
        result=analysis.report(a.plan_dir,a.output)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
