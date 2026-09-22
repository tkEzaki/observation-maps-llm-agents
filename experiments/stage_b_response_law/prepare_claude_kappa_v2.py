"""One-time, offline protocol amendment after syntax-only live preflight failure."""
from pathlib import Path
import json
import hashlib
import importlib

ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).parent
source=(D/'run_claude_kappa.py').read_text(encoding='utf-8')
source=source.replace('protocol_claude_kappa_v1.json','protocol_claude_kappa_v2.json')
source=source.replace("'runs/claude_kappa_v1'","'runs/claude_kappa_v2'")
source=source.replace('from circlemap.response import parse_social_action',
    'from circlemap.response import parse_social_action as parse_strict_social_action\nimport re\n\n'
    'def parse_social_action(raw_text):\n'
    '    text = raw_text.strip()\n'
    '    match = re.fullmatch(r"```(?:json)?\\s*\\n(.*?)\\n```", text, flags=re.DOTALL)\n'
    '    return parse_strict_social_action(match.group(1) if match else raw_text)\n')
source=source.replace("action = parse_social_action(result['raw_response'])", """try:
            parse_strict_social_action(result['raw_response'])
            result['strict_valid'] = True
        except (ValueError, TypeError):
            result['strict_valid'] = False
        action = parse_social_action(result['raw_response'])
        result['fence_removed'] = not result['strict_valid']""")
source=source.replace("'first_attempt_valid_rate':first_valid/len(tasks),", """'first_attempt_valid_rate':first_valid/len(tasks),
        'accepted_with_fence_removed':sum(bool(r.get('fence_removed')) for r in accepted.values()),
        'first_attempt_strict_valid_rate':sum(bool(r.get('strict_valid')) for (_,a),r in results.items() if a==1)/len(tasks),""")
target=D/'run_claude_kappa_v2.py'
if target.exists():
    raise RuntimeError('Amendment already prepared; refusing overwrite')
target.write_text(source,encoding='utf-8')
spec=json.loads((D/'protocol_claude_kappa_v1.json').read_text())
spec['experiment_id']='claude-kappa-v2'
spec['status']='syntax_amendment_after_9_preflight_responses_before_full_acquisition'
spec['invalid_policy']['parser']='Remove only one complete outer markdown JSON fence, then require the original strict one-field JSON schema. Prose, extra keys and invalid labels remain invalid.'
spec['invalid_policy']['macro_difference']='Matches the fenced-JSON subset of macro rescue; no regex extraction of labels from prose. Strict validity and fence removal are separately recorded.'
spec['amendment']={
    'date':'2026-09-18',
    'reason':'All 9 v1 preflight responses were JSON enclosed in markdown fences.',
    'prompt_unchanged':True,
    'reuse':'Import only the first received response for each preflight task. Later duplicate retries remain in the immutable v1 journal, excluded from analysis.',
    'interpretation':'Syntax change after inspecting raw responses, not an externally preregistered rule.',
    'parent_journal':'runs/claude_kappa_v1/anthropic/block_1/attempts.jsonl'}
(D/'protocol_claude_kappa_v2.json').write_text(json.dumps(spec,indent=2)+'\n')
runner=importlib.import_module('experiments.stage_b_response_law.run_claude_kappa_v2')
root=ROOT/'runs/claude_kappa_v2/anthropic'
with runner.run_lock(root):
    for b in ['block_1','block_2']:
        tasks,frozen=runner.make_plan(spec,b,'anthropic')
        folder=root/b;folder.mkdir(exist_ok=True)
        runner.initialize(folder,frozen,tasks)
        if b=='block_1':
            original=ROOT/spec['amendment']['parent_journal']
            lines=[json.loads(x) for x in original.read_text().splitlines()]
            firststarts={r['task_id']:r for r in lines if r['event']=='start' and r['attempt']==1}
            firstresults=[r for r in lines if r['event']=='result' and r['attempt']==1]
            assert len(firstresults)==len(firststarts)
            with (folder/'attempts.jsonl').open('x',encoding='utf-8') as handle:
                for row in firstresults:
                    parsed=runner.parse_social_action(row['raw_response'])
                    row.update(valid=True,action_label=parsed.label,action_value=parsed.value,
                               strict_valid=False,fence_removed=True,imported_from_v1=True)
                    row.pop('error_type',None);row.pop('http_status',None)
                    runner.append(handle,firststarts[row['task_id']])
                    runner.append(handle,row)
            runner.atomic_json(root/'preflight_import.json',{
                'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                'original_results':sum(r['event']=='result' for r in lines),
                'imported_first_responses':len(firstresults),
                'excluded_duplicate_responses':sum(r['event']=='result' for r in lines)-len(firstresults),
                'original_usage_input':sum(r.get('input_tokens',0) for r in lines),
                'original_usage_output':sum(r.get('output_tokens',0) for r in lines)})
        print(b,runner.export(folder,tasks,frozen))
