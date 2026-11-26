"""Simple end-to-end gold-set evaluator. Requires a running API and indexed sample document."""
import json, os, requests
API = os.getenv('API_URL', 'http://localhost:8000').rstrip('/')
with open('gold_qa.json', encoding='utf-8') as f: cases = json.load(f)
answer_hits = citation_hits = 0
for i, case in enumerate(cases, 1):
    r = requests.post(f'{API}/api/query', json={'question': case['question'], 'top_k': 12}, timeout=180)
    r.raise_for_status(); data = r.json()
    answer = data['answer'].lower()
    answer_ok = all(term.lower() in answer for term in case['expected_terms'])
    citation_ok = any(c['source'].endswith(case['expected_source']) for c in data['citations'])
    answer_hits += answer_ok; citation_hits += citation_ok
    print(f"{i}. answer={'PASS' if answer_ok else 'FAIL'} citation={'PASS' if citation_ok else 'FAIL'} | {case['question']}")
print(f'Answer exact-term success: {answer_hits}/{len(cases)} = {answer_hits/len(cases):.1%}')
print(f'Expected-source citation rate: {citation_hits}/{len(cases)} = {citation_hits/len(cases):.1%}')
