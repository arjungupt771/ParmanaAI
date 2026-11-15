import os, importlib

os.environ['LLM_PROVIDER'] = 'unsupported-provider'
import backend.agents.verifier as v
importlib.reload(v)

claim = 'A causes B'
evidence = [{'text': 'A causes B according to research', 'source': 'paper1'}]
res = v.verify_claim(claim, evidence)
print('verify_claim verdict:', res['verdict'])
if res['verdict'] != 'SUPPORTED':
    raise SystemExit(1)
print('All checks passed')
