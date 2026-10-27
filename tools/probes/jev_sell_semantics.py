"""Command-line entry point for the paired SELL semantics study."""
import argparse, asyncio, json
from capclaims.sell_semantics import runner

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('preview','freeze','run','replay','report')); args=p.parse_args()
    if args.action=='preview':
        old, schedule, bodies=asyncio.run(runner.preview())
        result={'paths':len(schedule),'max_body_bytes':max(b['bytes'] for b in bodies.values())}
    elif args.action=='freeze': result=asyncio.run(runner.freeze())
    elif args.action=='run': result=asyncio.run(runner.run())
    elif args.action=='replay':
        plan,predictions,status=asyncio.run(runner.replay()); result={'status':status,'predictions':len(predictions)}
    else:
        from capclaims.sell_semantics import analysis
        result=analysis.report()
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
