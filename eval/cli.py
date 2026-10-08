"""
Penny Evaluation CLI Tool
Usage:
  python eval/cli.py list [--filter VERDICT] [--limit N]
  python eval/cli.py inspect <trace_id>
  python eval/cli.py diagnose [<trace_id> | --all] [--llm]
  python eval/cli.py stats
  python eval/cli.py export [--csv]
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.tracker import (
    get_traces,
    get_trace,
    get_statistics,
    export_traces_data
)
from eval.evaluator import (
    evaluate_trace,
    evaluate_all_pending
)

VERDICT_COLORS = {
    'SUCCESS': '\033[92m',          # Green
    'RETRIEVAL_ISSUE': '\033[93m',  # Yellow
    'DATA_ISSUE': '\033[94m',       # Blue
    'MODEL_ISSUE': '\033[91m',      # Red
    'LANGUAGE_ISSUE': '\033[95m',   # Magenta
    'PENDING': '\033[90m',          # Gray
}
RESET = '\033[0m'
BOLD = '\033[1m'


def colorize_verdict(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, '\033[0m')
    return f"{color}{verdict}{RESET}"


def cmd_stats(args):
    stats = get_statistics()
    print(f"\n{BOLD}══════════════════════════════════════════════{RESET}")
    print(f"{BOLD}       PENNY EVALUATION & TRACING STATS       {RESET}")
    print(f"{BOLD}══════════════════════════════════════════════{RESET}")
    print(f"Total Conversations Logged : {stats['total_traces']}")
    print(f"Evaluated Interactions     : {stats['evaluated_total']}")
    print(f"Pending Evaluation         : {stats['pending_count']}")
    print(f"Average Total Latency      : {stats['avg_latency_ms']} ms")
    print(f"Average Retrieval Score    : {stats['avg_retrieval_score']}")
    print(f"Overall Success Rate       : {stats['success_rate_pct']}%\n")

    print(f"{BOLD}Breakdown by Root Cause:{RESET}")
    print(f"  {colorize_verdict('SUCCESS')}           : {stats['success_count']}")
    print(f"  {colorize_verdict('RETRIEVAL_ISSUE')}   : {stats['retrieval_issue_count']} (missed context in data/)")
    print(f"  {colorize_verdict('DATA_ISSUE')}        : {stats['data_issue_count']} (missing knowledge in corpus)")
    print(f"  {colorize_verdict('MODEL_ISSUE')}       : {stats['model_issue_count']} (hallucination or refusal)")
    print(f"  {colorize_verdict('LANGUAGE_ISSUE')}    : {stats['language_issue_count']} (script/language violation)")
    print(f"{BOLD}══════════════════════════════════════════════{RESET}\n")


def cmd_list(args):
    traces = get_traces(verdict_filter=args.filter, limit=args.limit, search=args.search)
    if not traces:
        print("No traces found matching criteria.")
        return

    print(f"\n{BOLD}{'ID (Short)':<10} {'Time':<12} {'Lang':<6} {'Score':<6} {'Verdict':<18} {'Query':<40}{RESET}")
    print("─" * 96)
    for t in traces:
        t_id = t['id'][:8]
        ts = t['timestamp'][11:19] if len(t['timestamp']) >= 19 else t['timestamp']
        lang = "EN" if t['language_code'] == '1' else "NE"
        score = f"{t['retrieval_score']:.1f}" if t.get('retrieval_score') is not None else "0.0"
        verdict = colorize_verdict(t['verdict'])
        query = (t['user_query'][:37] + '...') if len(t['user_query']) > 40 else t['user_query']
        print(f"{t_id:<10} {ts:<12} {lang:<6} {score:<6} {verdict:<27} {query:<40}")
    print(f"\nTotal shown: {len(traces)} traces. Use 'python eval/cli.py inspect <id>' for full details.\n")


def cmd_inspect(args):
    traces = get_traces(limit=500)
    target = None
    for t in traces:
        if t['id'].startswith(args.trace_id):
            target = t
            break

    if not target:
        print(f"Trace starting with '{args.trace_id}' not found.")
        return

    print(f"\n{BOLD}══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}TRACE DETAILS: {target['id']}{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════{RESET}")
    print(f"Timestamp   : {target['timestamp']}")
    print(f"Language    : {'English (1)' if target['language_code'] == '1' else 'Nepali (0)'}")
    print(f"Latency     : {target.get('total_latency_ms', 0):.1f} ms (Retrieval: {target.get('retrieval_latency_ms', 0):.1f} ms, LLM: {target.get('llm_latency_ms', 0):.1f} ms)")
    print(f"Status      : {target.get('status')}")
    print(f"\n{BOLD}USER QUERY:{RESET}\n  {target['user_query']}\n")

    chunks = target.get('retrieval_chunks') or []
    print(f"{BOLD}RETRIEVED CHUNKS ({len(chunks)}):{RESET}")
    if not chunks:
        print("  (None retrieved)")
    else:
        for idx, c in enumerate(chunks, 1):
            score = c.get('score', 0.0)
            print(f"  {idx}. [{c.get('source')}] {c.get('title')} (score: {score:.2f})")
            content_preview = c.get('content', '').replace('\n', ' ')[:140]
            print(f"     Preview: {content_preview}...")

    print(f"\n{BOLD}PENNY'S RESPONSE:{RESET}\n  {target.get('model_response', '(Empty)')}\n")

    print(f"{BOLD}EVALUATION VERDICT:{RESET} {colorize_verdict(target.get('verdict', 'PENDING'))}")
    if target.get('diagnosis_reason'):
        print(f"{BOLD}Reason:{RESET}\n  {target['diagnosis_reason']}")
    if target.get('suggested_fix'):
        print(f"{BOLD}Suggested Fix:{RESET}\n  {target['suggested_fix']}")
    if target.get('human_verdict'):
        print(f"{BOLD}Human Override:{RESET} {target['human_verdict']} ({target.get('human_notes', '')})")
    print(f"{BOLD}══════════════════════════════════════════════════════════════{RESET}\n")


def cmd_diagnose(args):
    use_llm = args.llm
    if args.all:
        print(f"Running batch diagnosis on all pending traces (LLM judge={use_llm})...")
        res = evaluate_all_pending(use_llm_judge=use_llm)
        print(f"Evaluated {res['total_evaluated']} traces.")
        for k, v in res['breakdown'].items():
            print(f"  {colorize_verdict(k)}: {v}")
    elif args.trace_id:
        # Find match by prefix
        traces = get_traces(limit=500)
        matched_id = None
        for t in traces:
            if t['id'].startswith(args.trace_id):
                matched_id = t['id']
                break
        if not matched_id:
            print(f"Trace not found: {args.trace_id}")
            return
        print(f"Diagnosing trace {matched_id} (LLM judge={use_llm})...")
        res = evaluate_trace(matched_id, use_llm_judge=use_llm)
        print(f"\nVerdict       : {colorize_verdict(res.get('verdict'))}")
        print(f"Confidence    : {res.get('confidence', 1.0)}")
        print(f"Reason        : {res.get('reason')}")
        print(f"Suggested Fix : {res.get('suggested_fix')}\n")
    else:
        print("Please specify a trace ID or use --all to diagnose all pending traces.")


def cmd_export(args):
    fmt = "csv" if args.csv else "json"
    data = export_traces_data(format=fmt)
    filename = f"eval_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)
    print(f"Exported evaluation traces to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Penny Evaluation & Diagnostic CLI")
    subparsers = parser.add_subparsers(dest="command")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show evaluation summary metrics")

    # list
    p_list = subparsers.add_parser("list", help="List recent chat traces")
    p_list.add_argument("--filter", help="Filter by verdict (SUCCESS, RETRIEVAL_ISSUE, DATA_ISSUE, etc.)")
    p_list.add_argument("--limit", type=int, default=25, help="Number of traces to show")
    p_list.add_argument("--search", help="Search query or response substring")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a specific trace by ID")
    p_inspect.add_argument("trace_id", help="Full or prefix trace ID")

    # diagnose
    p_diag = subparsers.add_parser("diagnose", help="Run root-cause diagnosis on a trace or all pending")
    p_diag.add_argument("trace_id", nargs="?", help="Trace ID to diagnose")
    p_diag.add_argument("--all", action="store_true", help="Diagnose all pending traces")
    p_diag.add_argument("--llm", action="store_true", help="Use local LLM judge instead of heuristic")

    # export
    p_exp = subparsers.add_parser("export", help="Export traces to file")
    p_exp.add_argument("--csv", action="store_true", help="Export in CSV format instead of JSON")

    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
