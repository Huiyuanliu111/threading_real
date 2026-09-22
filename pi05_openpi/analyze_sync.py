"""Summarize --sync-diagnostics JSONL without contacting hardware."""
import argparse
import json
from collections import defaultdict


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('cycle', 0):
            groups[(row.get('session'), row['episode'], row['cycle'])].append(row)
    result = []
    for (session, episode, cycle), events in groups.items():
        by_event = {r['event']: r for r in events}
        out = dict(session=session, episode=episode, cycle=cycle)
        for label, start, end in [('camera_read_ms','camera_read_start','camera_read_end'),
                                  ('inference_ms','inference_start','inference_end')]:
            if start in by_event and end in by_event:
                out[label] = 1000*(by_event[end]['monotonic_s']-by_event[start]['monotonic_s'])
        if 'inference_start' in by_event and 'inference_end' in by_event:
            before=by_event['inference_start'].get('sender', {})
            after=by_event['inference_end'].get('sender', {})
            if 'successful_send_count' in before and 'successful_send_count' in after:
                out['udp_sends_during_inference']=after['successful_send_count']-before['successful_send_count']
        origin = by_event.get('execution_origin', {})
        if 'observation_to_execution_translation_m' in origin:
            out['observation_to_execution_net_motion_mm'] = 1000*origin['observation_to_execution_translation_m']
            out['observation_to_execution_net_rotation_deg'] = origin['observation_to_execution_rotation_rad']*180/3.141592653589793
        wait = [e for e in events if e['event']=='wait_sample']
        complete = [e for e in wait if e['segment_completed']]
        for name, samples in [('first_send_completed', complete[:1]), ('last_wait', wait[-1:])]:
            if samples:
                e=samples[0]
                out[name]={'error_mm':1000*e['translation_error_m'],
                           'error_deg':e['rotation_error_rad']*180/3.141592653589793,
                           'max_abs_dq_rad_s':max(map(abs,e.get('state',{}).get('dq',[])),default=None),
                           'sender':e.get('sender')}
        if complete and wait:
            out['observed_hold_after_send_s']=wait[-1]['monotonic_s']-complete[0]['monotonic_s']
        out['timeout']='wait_timeout' in by_event
        result.append(out)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path');a=p.parse_args()
    with open(a.path) as f: rows=[json.loads(line) for line in f if line.strip()]
    for result in summarize(rows):print(json.dumps(result,ensure_ascii=False))
