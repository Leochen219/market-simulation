"""Offline preview/freeze, run, replay and aggregate analysis for expanded study."""
import argparse,asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from capclaims.expanded_grid import runner,analysis

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('preview','freeze','run','replay','report'));p.add_argument('--plan-dir',type=Path,default=runner.OUT);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.command=='preview':
        contexts,targets,metadata,size=asyncio.run(runner.preview()); result={'contexts':len(contexts),'paths':len(__import__('capclaims.expanded_grid.design',fromlist=['schedule']).schedule(contexts)),'max_body_bytes':size}
    elif args.command=='freeze': result=asyncio.run(runner.freeze(args.plan_dir))
    elif args.command=='run': result=asyncio.run(runner.run(args.plan_dir))
    elif args.command=='replay':
        _,pred,status=runner.replay(args.plan_dir); result={'status':status,'replayed_predictions':len(pred)}
    else:
        if args.output is None: p.error('--output required for report')
        result=analysis.report(args.plan_dir,args.output)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
