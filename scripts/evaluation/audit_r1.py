from pathlib import Path
import json, re, shlex, importlib.util, statistics as st, math, hashlib
from collections import Counter, defaultdict
lab=Path.cwd(); r=lab/'experiments/R1_grpo_baseline'
man=json.loads((r/'run_manifest.json').read_text())
log=Path(man['trainer_log']).read_text(errors='replace')
line=next(x for x in log.splitlines() if x.startswith('+ uv run ') and 'verl.trainer.main_ppo' in x)
args=shlex.split(line[2:]); overrides=dict(x.lstrip('+').split('=',1) for x in args[args.index('verl.trainer.main_ppo')+1:] if '=' in x)
(r/'launch_overrides.json').write_text(json.dumps({'source':'actual shell xtrace in trainer.log','overrides':overrides},indent=2))
cap=int(overrides['data.max_response_length'])
spec=importlib.util.spec_from_file_location('math_dapo_audit',lab/'repos/verl/verl/utils/reward_score/math_dapo.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.compute_score('Answer: 5','5')['score']==1
assert m.compute_score('Answer: 4','5')['score']==-1
assert m.compute_score('no answer','5')['pred']=='[INVALID]'
assert m.compute_score('Answer: 5\n'+'x'*310,'5')['pred']=='[INVALID]'
records=[json.loads(x) for x in (r/'metrics/verl_file_logger.jsonl').read_text().splitlines()]
assert [x['step'] for x in records]==list(range(1,21))
out=[]; prompts=Counter(); samples=[]; total_states=Counter()
for rec in records:
 step=rec['step']; d=rec['data']
 p=Path(man['rollout_dump_dir'])/f'{step}.jsonl'
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 assert len(rows)==128
 groups=defaultdict(list); states=Counter(); mismatch=0; excerpt=[]
 for row in rows:
  groups[row['input']].append(row)
  scored=m.compute_score(row['output'],str(row['gts']))
  mismatch+=scored['score']!=row['score']
  state='PARSE_FAILURE' if scored['pred']=='[INVALID]' else ('VERIFIER_CORRECT' if scored['acc'] else 'PARSED_NOT_MATCHED')
  states[state]+=1
  if step in [10,14,18] and len(excerpt)<2 and row['score']==-1:
   excerpt.append({'uid':row['uid'],'gt':row['gts'],'pred':scored['pred'],'tail':row['output'][-350:]})
 assert len(groups)==16 and all(len(v)==8 for v in groups.values())
 assert abs(st.mean(x['score'] for x in rows)-d['critic/rewards/mean'])<1e-9
 hist=Counter(sum(v['score']==1 for v in g) for g in groups.values())
 prompts.update(hashlib.sha256(k.encode()).hexdigest() for k in groups)
 mx=d['response_length/max']; cap_rate=0.0 if mx<cap else d['response_length/clip_ratio'] if mx==cap else None
 total_states.update(states)
 out.append({'step':step,'reward':d['critic/rewards/mean'],'states':dict(states),'rescore_mismatch':mismatch,'correct_per_group_hist':dict(hist),'all_correct':hist[8],'all_wrong':hist[0],'mixed':16-hist[0]-hist[8], 'effective_signal_fraction':(16-hist[0]-hist[8])/16,'response_mean':d['response_length/mean'],'response_max':mx,'raw_width_hit_fraction':d['response_length/clip_ratio'],'cap_hit_rate':cap_rate,'cap_hit_count':round(cap_rate*128) if cap_rate is not None else None,'rollout_tokens_from_native_mean':round(d['response_length/mean']*128), 'examples':excerpt})
 for row in sorted(rows,key=lambda x:len(x['output']),reverse=True)[:1] if step in [10,14,18] else []:
  samples.append({'step':step,'uid':row['uid'],'score':row['score'],'gt':row['gts'],'characters':len(row['output']),'head':row['output'][:300],'tail':row['output'][-900:]})
def summary(vals):
 return {'first':vals[0],'last':vals[-1],'mean':st.mean(vals),'min':min(vals),'max':max(vals),'first5':st.mean(vals[:5]),'last5':st.mean(vals[-5:]),'first10':st.mean(vals[:10]),'last10':st.mean(vals[-10:])}
keys=['critic/rewards/mean','actor/entropy','actor/pg_clipfrac','actor/ppo_kl','actor/grad_norm','actor/kl_loss','response_length/mean','response_length/max','timing_s/gen','timing_s/old_log_prob','timing_s/ref','timing_s/update_actor','timing_s/update_weights','timing_s/step']
stats={k:summary([x['data'][k] for x in records]) for k in keys}
times={k:sum(x['data'].get(k,0) for x in records) for k in ['timing_s/gen','timing_s/old_log_prob','timing_s/ref','timing_s/update_actor','timing_s/update_weights','timing_s/step','timing_s/testing','timing_s/adv']}
audit={'run_uid':man['run_uid'],'verifier_sha256':hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),'launch_overrides':overrides,'rows':out,'stats':stats,'timing_sums':times,'states_total':dict(total_states),'unique_prompts':len(prompts),'prompt_appearances':sum(prompts.values()),'repeated_prompts':sum(v>1 for v in prompts.values()),'n_rollouts':sum(sum(x['states'].values()) for x in out),'cap_hit_count':sum(x['cap_hit_count'] for x in out),'output_tokens_native':sum(x['rollout_tokens_from_native_mean'] for x in out),'total_tokens_including_prompts':sum(x['data']['perf/total_num_tokens'] for x in records),'wall_seconds':man['end_ts']-man['start_ts'],'nonfinite':[(x['step'],k) for x in records for k,v in x['data'].items() if isinstance(v,(int,float)) and not math.isfinite(v)],'staleness_max':max(x['data']['training/off_policy/trajectory_staleness/max'] for x in records),'rescore_mismatches':sum(x['rescore_mismatch'] for x in out),'longest_examples':samples}
(lab/'analysis/R1_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k not in ['launch_overrides','rows','longest_examples']},indent=2))
print('INCIDENT_STEPS',json.dumps([x for x in out if x['step'] in [10,14,18]],indent=2))
print('LONGEST',json.dumps(samples,indent=2))
