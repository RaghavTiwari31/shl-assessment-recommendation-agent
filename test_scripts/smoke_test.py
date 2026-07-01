import requests
import time

BASE = 'http://localhost:8000'

def chat(messages, timeout=60):
    r = requests.post(f'{BASE}/chat', json={'messages': messages}, timeout=timeout)
    return r.json()

def show(label, r):
    recs = r.get('recommendations', [])
    reply = r['reply'][:160]
    eoc = r['end_of_conversation']
    print(f'[{label}]')
    print(f'  reply: {reply}')
    print(f'  recs: {len(recs)} | eoc: {eoc}')
    for rec in recs[:3]:
        name = rec['name']
        tt = rec['test_type']
        print(f'    - {name} | {tt}')
    print()

print('--- Starting tests ---\n')

show('CLARIFY', chat([{'role': 'user', 'content': 'I need an assessment'}]))
time.sleep(15)

show('RECOMMEND', chat([
    {'role': 'user', 'content': 'I am hiring a mid-level Java backend developer. Need to assess Java programming and software design skills.'},
]))
time.sleep(15)

show('REFUSE', chat([{'role': 'user', 'content': 'What salary should I pay a Python developer?'}]))
time.sleep(15)

show('COMPARE', chat([{'role': 'user', 'content': 'Compare the OPQ32r and the Global Skills Assessment'}]))
time.sleep(15)

show('REFINE', chat([
    {'role': 'user', 'content': 'I need assessments for a sales manager'},
    {'role': 'assistant', 'content': 'I recommend OPQ32r, Sales Transformation Report 1.0, HiPo Assessment Report.'},
    {'role': 'user', 'content': 'Only include personality tests please'}
]))
