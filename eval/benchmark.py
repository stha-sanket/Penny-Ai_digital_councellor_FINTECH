"""
Penny Benchmark Runner
Executes benchmark test cases from eval/test_cases.json against Penny,
records traces, runs root cause diagnostics, and prints evaluation report.
"""

import os
import sys
import json
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.tracker import get_trace
from eval.evaluator import evaluate_trace

TEST_CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_cases.json')
SERVER_URL = os.environ.get('PENNY_SERVER_URL', 'http://localhost:5000')

BOLD = '\033[1m'
RESET = '\033[0m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'


def run_benchmark(limit: int = None, use_llm_judge: bool = False):
    if not os.path.exists(TEST_CASES_FILE):
        print(f"Test cases file not found: {TEST_CASES_FILE}")
        return

    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    if limit:
        cases = cases[:limit]

    print(f"\n{BOLD}══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}         RUNNING PENNY EVALUATION BENCHMARK SUITE             {RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════{RESET}")
    print(f"Total Test Cases: {len(cases)}")
    print(f"Target Server   : {SERVER_URL}\n")

    results = []

    for idx, tc in enumerate(cases, 1):
        query = tc['query']
        lang = tc.get('lang', '1')
        print(f"[{idx}/{len(cases)}] Query: \"{query}\" (Lang: {'EN' if lang == '1' else 'NE'})")

        # 1. Ensure server language is set
        try:
            requests.post(f"{SERVER_URL}/api/language", json={'language': lang}, timeout=5)
        except Exception as e:
            print(f"  ❌ Cannot connect to Penny server at {SERVER_URL}. Is it running?")
            return

        # 2. Call /api/chat
        start_t = time.time()
        try:
            res = requests.post(
                f"{SERVER_URL}/api/chat",
                json={'message': query, 'stream': False},
                timeout=45
            )
            elapsed = time.time() - start_t
            if res.status_code != 200:
                print(f"  ❌ Server error: {res.status_code}")
                continue

            chat_data = res.json()
            response_text = chat_data.get('response', '')
            # The server logs traces automatically, let's grab the latest trace id from headers or query traces
            trace_id = res.headers.get('X-Penny-Trace-Id')
            
            # If header not present, we can look up trace via /api/eval/traces
            if not trace_id:
                eval_res = requests.get(f"{SERVER_URL}/api/eval/traces?limit=1").json()
                if eval_res and len(eval_res) > 0:
                    trace_id = eval_res[0]['id']

            if not trace_id:
                print("  ⚠️ Could not resolve trace ID")
                continue

            # 3. Diagnose trace
            diag = evaluate_trace(trace_id, use_llm_judge=use_llm_judge)
            verdict = diag.get('verdict', 'PENDING')
            expected = tc.get('expected_verdict')

            is_match = (verdict == expected) or (expected == 'SUCCESS' and verdict == 'SUCCESS')

            verdict_color = GREEN if verdict == 'SUCCESS' else (YELLOW if verdict == 'RETRIEVAL_ISSUE' else BLUE)
            print(f"  Response : {response_text[:80]}...")
            print(f"  Verdict  : {verdict_color}{verdict}{RESET} (Expected: {expected}) -> {'✅ MATCH' if is_match else '⚠️ DEVIATION'}")
            print(f"  Reason   : {diag.get('reason', '')[:100]}...\n")

            results.append({
                'case': tc,
                'verdict': verdict,
                'expected': expected,
                'matched': is_match,
                'latency_s': round(elapsed, 2),
                'response': response_text,
                'reason': diag.get('reason', '')
            })

        except Exception as e:
            print(f"  ❌ Error executing test case: {e}\n")

    # Summary
    if results:
        total = len(results)
        matches = sum(1 for r in results if r['matched'])
        accuracy = round(matches / total * 100, 1)

        retrieval_issues = sum(1 for r in results if r['verdict'] == 'RETRIEVAL_ISSUE')
        data_issues = sum(1 for r in results if r['verdict'] == 'DATA_ISSUE')
        model_issues = sum(1 for r in results if r['verdict'] == 'MODEL_ISSUE')
        lang_issues = sum(1 for r in results if r['verdict'] == 'LANGUAGE_ISSUE')
        successes = sum(1 for r in results if r['verdict'] == 'SUCCESS')

        print(f"\n{BOLD}══════════════════════════════════════════════════════════════{RESET}")
        print(f"{BOLD}                    BENCHMARK RESULTS                         {RESET}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{RESET}")
        print(f"Total Cases Evaluated : {total}")
        print(f"Expected Alignment    : {matches}/{total} ({accuracy}%)")
        print(f"\nVerdict Distribution:")
        print(f"  ✅ SUCCESS          : {successes}")
        print(f"  🔍 RETRIEVAL_ISSUE  : {retrieval_issues}")
        print(f"  📁 DATA_ISSUE       : {data_issues}")
        print(f"  🧠 MODEL_ISSUE      : {model_issues}")
        print(f"  🌐 LANGUAGE_ISSUE   : {lang_issues}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{RESET}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Penny Evaluation Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--llm", action="store_true", help="Use local LLM judge")
    args = parser.parse_args()
    run_benchmark(limit=args.limit, use_llm_judge=args.llm)
