"""
Penny Evaluation & Tracing Engine (Tracker)
- Logs every chat turn: user query, retrieved context, scores, prompt, response, latency
- Stores traces in SQLite database (eval/eval_traces.db) with JSONL fallback
- Manages diagnostics, verdict updates, and human annotations
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_traces.db')
JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_traces.jsonl')


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Initializes the database schema if not already created."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_query TEXT NOT NULL,
                language_code TEXT NOT NULL,
                detected_lang_action TEXT,
                is_greeting INTEGER NOT NULL,
                retrieval_chunks TEXT,
                retrieval_score REAL,
                num_retrieved INTEGER,
                system_prompt TEXT,
                model_response TEXT,
                model_name TEXT,
                stream_requested INTEGER,
                retrieval_latency_ms REAL,
                llm_latency_ms REAL,
                total_latency_ms REAL,
                status TEXT,
                verdict TEXT DEFAULT 'PENDING',
                diagnosis_reason TEXT,
                suggested_fix TEXT,
                diagnosis_details TEXT,
                human_verdict TEXT,
                human_notes TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_verdict ON traces(verdict)")
    conn.close()


init_db()


def log_chat_interaction(
    user_query: str,
    language_code: str,
    retrieval_chunks: List[Dict[str, Any]],
    system_prompt: str,
    model_response: str = "",
    model_name: str = "gemma4:e2b",
    session_id: Optional[str] = None,
    detected_lang_action: Optional[str] = None,
    is_greeting: bool = False,
    stream_requested: bool = False,
    retrieval_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    total_latency_ms: float = 0.0,
    status: str = "success",
    trace_id: Optional[str] = None
) -> str:
    """
    Logs an interaction into the SQLite traces database and returns the trace_id.
    """
    t_id = trace_id or str(uuid.uuid4())
    ts = datetime.utcnow().isoformat() + "Z"
    
    total_score = sum(c.get('score', 0.0) for c in retrieval_chunks) if retrieval_chunks else 0.0
    num_retrieved = len(retrieval_chunks)

    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces (
                    id, timestamp, session_id, user_query, language_code,
                    detected_lang_action, is_greeting, retrieval_chunks,
                    retrieval_score, num_retrieved, system_prompt,
                    model_response, model_name, stream_requested,
                    retrieval_latency_ms, llm_latency_ms, total_latency_ms,
                    status, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t_id, ts, session_id, user_query, language_code,
                detected_lang_action, 1 if is_greeting else 0,
                json.dumps(retrieval_chunks, ensure_ascii=False),
                total_score, num_retrieved, system_prompt,
                model_response, model_name, 1 if stream_requested else 0,
                retrieval_latency_ms, llm_latency_ms, total_latency_ms,
                status, 'PENDING'
            ))
    except Exception as e:
        print(f"⚠️ Error logging trace {t_id} to DB: {e}")
    finally:
        conn.close()

    # Fallback/convenience JSONL log
    try:
        entry = {
            "id": t_id,
            "timestamp": ts,
            "user_query": user_query,
            "language_code": language_code,
            "is_greeting": is_greeting,
            "num_retrieved": num_retrieved,
            "retrieval_score": total_score,
            "model_response": model_response,
            "status": status,
        }
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Error appending to JSONL log: {e}")

    return t_id


def update_trace_response(
    trace_id: str,
    model_response: str,
    llm_latency_ms: float,
    total_latency_ms: float,
    status: str = "success"
):
    """Updates model response and timing for streaming or deferred completions."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE traces
                SET model_response = ?,
                    llm_latency_ms = ?,
                    total_latency_ms = ?,
                    status = ?
                WHERE id = ?
            """, (model_response, llm_latency_ms, total_latency_ms, status, trace_id))
    except Exception as e:
        print(f"⚠️ Error updating trace {trace_id}: {e}")
    finally:
        conn.close()


def update_trace_diagnosis(
    trace_id: str,
    verdict: str,
    diagnosis_reason: str,
    suggested_fix: str,
    diagnosis_details: Dict[str, Any]
):
    """Saves evaluation diagnosis results to a trace."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE traces
                SET verdict = ?,
                    diagnosis_reason = ?,
                    suggested_fix = ?,
                    diagnosis_details = ?
                WHERE id = ?
            """, (
                verdict,
                diagnosis_reason,
                suggested_fix,
                json.dumps(diagnosis_details, ensure_ascii=False),
                trace_id
            ))
    except Exception as e:
        print(f"⚠️ Error updating diagnosis for {trace_id}: {e}")
    finally:
        conn.close()


def update_human_feedback(trace_id: str, human_verdict: str, human_notes: str):
    """Saves human review or override."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE traces
                SET human_verdict = ?,
                    human_notes = ?
                WHERE id = ?
            """, (human_verdict, human_notes, trace_id))
    except Exception as e:
        print(f"⚠️ Error updating human feedback for {trace_id}: {e}")
    finally:
        conn.close()


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single trace by ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get('retrieval_chunks'):
            try:
                d['retrieval_chunks'] = json.loads(d['retrieval_chunks'])
            except Exception:
                pass
        if d.get('diagnosis_details'):
            try:
                d['diagnosis_details'] = json.loads(d['diagnosis_details'])
            except Exception:
                pass
        return d
    finally:
        conn.close()


def get_traces(
    verdict_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetches traces with optional verdict filtering and search."""
    conn = get_db_connection()
    try:
        query = "SELECT * FROM traces WHERE 1=1"
        params: List[Any] = []

        if verdict_filter and verdict_filter.upper() != "ALL":
            query += " AND verdict = ?"
            params.append(verdict_filter.upper())

        if search:
            query += " AND (user_query LIKE ? OR model_response LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            if d.get('retrieval_chunks'):
                try:
                    d['retrieval_chunks'] = json.loads(d['retrieval_chunks'])
                except Exception:
                    pass
            if d.get('diagnosis_details'):
                try:
                    d['diagnosis_details'] = json.loads(d['diagnosis_details'])
                except Exception:
                    pass
            results.append(d)
        return results
    finally:
        conn.close()


def get_statistics() -> Dict[str, Any]:
    """Computes aggregate evaluation statistics across all traces."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM traces")
        total_count = cursor.fetchone()[0]

        cursor = conn.execute("""
            SELECT verdict, COUNT(*) as cnt
            FROM traces
            GROUP BY verdict
        """)
        verdict_counts = {row['verdict']: row['cnt'] for row in cursor.fetchall()}

        cursor = conn.execute("""
            SELECT 
                AVG(total_latency_ms) as avg_latency,
                AVG(retrieval_score) as avg_retrieval_score,
                SUM(CASE WHEN is_greeting = 1 THEN 1 ELSE 0 END) as greeting_count
            FROM traces
        """)
        row = cursor.fetchone()

        avg_latency = row['avg_latency'] or 0.0
        avg_retrieval_score = row['avg_retrieval_score'] or 0.0
        greeting_count = row['greeting_count'] or 0

        evaluated_total = sum(v for k, v in verdict_counts.items() if k != 'PENDING')
        success_count = verdict_counts.get('SUCCESS', 0)
        retrieval_issue_count = verdict_counts.get('RETRIEVAL_ISSUE', 0)
        data_issue_count = verdict_counts.get('DATA_ISSUE', 0)
        model_issue_count = verdict_counts.get('MODEL_ISSUE', 0)
        language_issue_count = verdict_counts.get('LANGUAGE_ISSUE', 0)

        success_rate = (success_count / evaluated_total * 100) if evaluated_total > 0 else 0.0

        return {
            "total_traces": total_count,
            "evaluated_total": evaluated_total,
            "pending_count": verdict_counts.get('PENDING', 0),
            "success_count": success_count,
            "retrieval_issue_count": retrieval_issue_count,
            "data_issue_count": data_issue_count,
            "model_issue_count": model_issue_count,
            "language_issue_count": language_issue_count,
            "success_rate_pct": round(success_rate, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_retrieval_score": round(avg_retrieval_score, 2),
            "verdict_breakdown": verdict_counts
        }
    finally:
        conn.close()


def export_traces_data(format: str = "json") -> str:
    """Exports all traces as JSON or CSV string."""
    traces = get_traces(limit=10000)
    if format.lower() == "csv":
        import csv
        import io
        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "user_query", "language_code", "is_greeting",
            "num_retrieved", "retrieval_score", "model_response", "verdict",
            "diagnosis_reason", "suggested_fix", "human_verdict"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in traces:
            writer.writerow(t)
        return output.getvalue()
    else:
        return json.dumps(traces, indent=2, ensure_ascii=False)
