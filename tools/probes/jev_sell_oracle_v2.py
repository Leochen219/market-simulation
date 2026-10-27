"""Offline preview and one-shot lifecycle for the SELL oracle diagnostic."""
import argparse,asyncio,json
from capclaims.sell_oracle import runner_v2 as runner

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preview','freeze','run','replay','report'));args=p.parse_args()
    if args.action=='preview':
        plan,labels,bodies=asyncio.run(runner.preview());result={'contexts':len(plan['contexts']),'components':len(plan['components']),'queries':len(plan['queries']),'max_body_bytes':max(x['bytes'] for x in bodies.values()),'coverage':plan['metadata']}
    elif args.action=='freeze':result=asyncio.run(runner.freeze())
    elif args.action=='run':result=asyncio.run(runner.run())
    elif args.action=='replay':
        plan,predictions,status=asyncio.run(runner.replay());result={'status':status,'component_predictions':len(predictions)}
    else:
        from capclaims.sell_oracle import analysis_v2 as analysis
        result=analysis.report()
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
