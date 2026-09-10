"""CPU-only regression checks for metric semantics and missingness."""
import tempfile
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metric_collector as mc

def row(data, cap=None):
    return mc.build_row(1, data, None, 0, None, 'synthetic', {}, cap)

r = row({'response_length/max': 6625, 'response_length/clip_ratio': 1/128,
         'timing_s/step': 100, 'actor/pg_clipfrac': 0.002,
         'actor/pg_clipfrac_lower': 0.0003},8192)
assert r['cap_hit_rate'] == 0
assert r['truncation_rate'] is None
assert r['pct_reward'] is None
assert r['pct_logprob'] is None
assert r['clip_fraction'] == 0.002
assert r['dual_clip_fraction'] == 0.0003
assert r['clip_fraction_high'] is None and r['clip_fraction_low'] is None
assert row({'response_length/max':8192,'response_length/clip_ratio':3/128},8192)['cap_hit_rate']==3/128
assert row({'response_length/max':8192,'response_length/clip_ratio':3/128})['cap_hit_rate'] is None
assert row({'timing_s/step':100,'timing_s/reward':0})['pct_reward']==0
assert row({'val-core/math_dapo/acc/mean@1':0.495})['validation_accuracy']==0.495
assert row({})['validation_accuracy'] is None
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp); (p/'launch_overrides.json').write_text(json.dumps({'overrides':{'data.max_response_length':'8192'}}))
    sink=p/'sink.jsonl'; sink.write_text(json.dumps({'step':1,'data':{'response_length/max':7000,'response_length/clip_ratio':1/128}})+'\n')
    r=mc.collect(tmp,str(sink),None)
    assert len(r)==1 and r[0]['cap_hit_rate']==0
print('PASS: synthetic metric missingness, clipping semantics, cap proxy, validation, collect')
