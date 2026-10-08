"""
Penny Root-Cause Evaluator & Diagnostic Engine
Distinguishes between:
  1. RETRIEVAL_ISSUE: Knowledge is in data/*.md, but retriever missed it or returned irrelevant chunks.
  2. DATA_ISSUE: Information is missing from the entire data/ corpus (knowledge gap).
  3. MODEL_ISSUE: Context was provided, but model hallucinated, gave false refusal, or failed rules.
  4. LANGUAGE_ISSUE: Model replied in wrong language/script.
  5. SUCCESS: Accurate, grounded, and compliant answer.
"""

import os
import re
import glob
import json
import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import requests

from .tracker import (
    get_trace,
    update_trace_diagnosis
)

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/chat')
DEFAULT_MODEL = os.environ.get('PENNY_MODEL', 'gemma4:e2b')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Canonical refusal keywords
REFUSAL_PHRASES_EN = [
    "don't have information about that",
    "do not have information about that",
    "not found in my knowledge base",
    "cannot find information",
    "only help with questions about sunway",
    "no information available",
    "didn't quite get that",
]

REFUSAL_PHRASES_NE = [
    "जानकारी छैन",
    "ज्ञानकोषमा",
    "मेरो ज्ञानकोष",
    "सनवे कलेज काठमाडौंका कार्यक्रम",
    "बुझिन",
]

STOP_WORDS = frozenset([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if',
    'while', 'about', 'up', 'that', 'this', 'it', 'its', 'he', 'she',
    'they', 'them', 'we', 'you', 'i', 'me', 'my', 'your', 'his', 'her',
    'our', 'their', 'what', 'which', 'who', 'whom', 'these', 'those',
    'tell', 'me', 'give', 'know', 'please'
])


def stem(word: str) -> str:
    """Simple rule-based English suffix stripper for robust matching."""
    w = word.lower()
    for suffix in ('ation', 'ated', 'ating', 'tion', 'ment', 'ence', 'ance', 'ness', 'ers', 'ing', 'ies', 'es', 'ed', 'er', 's'):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def tokens_overlap(tokens: List[str], target_text: str) -> List[str]:
    """Finds query tokens that match words in target text either exactly or via root stem."""
    target_words = set(re.findall(r'[a-zA-Z0-9\u0900-\u097F]+', target_text.lower()))
    target_stems = {stem(w) for w in target_words}
    matched = []
    for t in tokens:
        t_stem = stem(t)
        if t in target_words or t_stem in target_stems or any(w.startswith(t_stem) for w in target_words if len(t_stem) >= 4):
            matched.append(t)
    return matched


class CorpusScanner:
    """
    Exhaustive scanner across all markdown files in data/ to verify if any
    section anywhere in the corpus contains answers to a user query.
    """
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.sections = []
        self._load_corpus()

    def _load_corpus(self):
        md_files = sorted(glob.glob(os.path.join(self.data_dir, '*.md')))
        self.sections = []
        for filepath in md_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                raw_sections = re.split(r'\n(?=#{1,3}\s)|(?:\n---\n)', content)
                for sec in raw_sections:
                    sec = sec.strip()
                    if not sec or len(sec) < 10:
                        continue
                    m = re.match(r'^#{1,5}\s+(.+)', sec)
                    title = m.group(1).strip() if m else filename
                    self.sections.append({
                        'source': filename,
                        'title': title,
                        'content': sec,
                        'lower_content': sec.lower()
                    })
            except Exception as e:
                print(f"⚠️ Error reading corpus file {filename}: {e}")

    def tokenize(self, text: str) -> List[str]:
        clean = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', ' ', text.lower())
        tokens = [t for t in clean.split() if t not in STOP_WORDS and len(t) > 1]
        return tokens

    def scan_for_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Scans all sections in the entire corpus and ranks them by token overlap
        and substring match. Returns matches with similarity score.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        results = []
        for sec in self.sections:
            content = sec['lower_content']
            matched_terms = tokens_overlap(query_tokens, content)
            if not matched_terms:
                continue

            score = len(matched_terms) / len(query_tokens)
            # Boost if query tokens appear in title
            title_lower = sec['title'].lower()
            if any(t in title_lower or stem(t) in title_lower for t in query_tokens):
                score += 0.5

            results.append({
                'source': sec['source'],
                'title': sec['title'],
                'content': sec['content'],
                'score': round(score, 3),
                'matched_terms': matched_terms
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results


corpus_scanner = CorpusScanner()


def is_refusal_response(text: str, lang_code: str) -> bool:
    """Checks if the response is a refusal/unknown canned reply."""
    lower = text.lower()
    phrases = REFUSAL_PHRASES_EN if lang_code == '1' else REFUSAL_PHRASES_NE
    for p in phrases:
        if p.lower() in lower:
            return True
    return False


def check_language_compliance(text: str, lang_code: str) -> Tuple[bool, str]:
    """
    Validates whether the model adhered to English vs Nepali Devanagari rules.
    """
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    latin_letters = len(re.findall(r'[a-zA-Z]', text))
    
    if lang_code == '0':  # Nepali required
        if not has_devanagari and latin_letters > 10:
            return False, "Failed Nepali rule: response was written in English/Latin alphabet instead of Devanagari."
        if not has_devanagari:
            return False, "Failed Nepali rule: missing Devanagari script."
    elif lang_code == '1':  # English required
        if has_devanagari:
            return False, "Failed English rule: response contained Devanagari script when English was requested."

    return True, "Language compliant."


def diagnose_trace_heuristic(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast rule-based root cause diagnostic engine.
    Analyzes:
      - Retrieved chunks vs whole corpus
      - Refusal status
      - Hallucination / Groundedness checks
      - Language constraints
    """
    user_query = trace.get('user_query', '').strip()
    lang_code = trace.get('language_code', '1')
    model_response = trace.get('model_response', '').strip()
    is_greeting = bool(trace.get('is_greeting', 0))
    retrieved_chunks = trace.get('retrieval_chunks') or []

    # 1. Greetings / Language switch commands
    if is_greeting:
        return {
            'verdict': 'SUCCESS',
            'confidence': 1.0,
            'reason': "Conversational greeting or farewell correctly handled without RAG.",
            'suggested_fix': "No action needed.",
            'details': {'type': 'greeting'}
        }

    if trace.get('detected_lang_action') or trace.get('status') == 'switch_command':
        return {
            'verdict': 'SUCCESS',
            'confidence': 1.0,
            'reason': "Language switch command successfully recognized and processed.",
            'suggested_fix': "No action needed.",
            'details': {'type': 'language_switch'}
        }

    # 2. Language constraint check
    lang_ok, lang_reason = check_language_compliance(model_response, lang_code)
    if not lang_ok:
        return {
            'verdict': 'LANGUAGE_ISSUE',
            'confidence': 0.95,
            'reason': f"Model violated language constraints: {lang_reason}",
            'suggested_fix': "Enforce strict system prompt instruction or check language prompt examples.",
            'details': {
                'language_expected': 'Nepali (Devanagari)' if lang_code == '0' else 'English',
                'response_snippet': model_response[:100]
            }
        }

    # 3. Analyze Retrieval
    has_retrieved_chunks = len(retrieved_chunks) > 0
    total_retrieval_score = sum(c.get('score', 0.0) for c in retrieved_chunks)
    
    query_tokens = corpus_scanner.tokenize(user_query)
    retrieved_texts = " ".join([c.get('content', '') + " " + c.get('title', '') for c in retrieved_chunks])

    # Common domain words that appear across almost all Sunway documents
    GENERIC_DOMAIN_WORDS = frozenset({'sunway', 'college', 'kathmandu', 'nepal', 'bcu', 'birmingham', 'university'})
    distinctive_query_tokens = [t for t in query_tokens if t not in GENERIC_DOMAIN_WORDS]
    if not distinctive_query_tokens:
        distinctive_query_tokens = query_tokens

    # Check what the full corpus scanner finds
    corpus_matches = corpus_scanner.scan_for_query(user_query)
    # Check if any corpus section matches the distinctive query terms
    best_corpus_match = None
    corpus_has_distinctive = False
    if corpus_matches:
        min_required = max(1, min(2, len(distinctive_query_tokens)))
        for m in corpus_matches:
            matched_distinctive = [t for t in distinctive_query_tokens if t in m['matched_terms'] or stem(t) in [stem(x) for x in m['matched_terms']]]
            if len(matched_distinctive) >= min_required and m['score'] >= 0.4:
                best_corpus_match = m
                corpus_has_distinctive = True
                break
        if not best_corpus_match:
            best_corpus_match = corpus_matches[0]

    is_refusal = is_refusal_response(model_response, lang_code)

    # 4. Scenario Analysis

    # Case A: Model refused to answer
    if is_refusal:
        # If distinctive tokens do not exist anywhere in the corpus, it's a DATA GAP
        if not corpus_has_distinctive:
            return {
                'verdict': 'DATA_ISSUE',
                'confidence': 0.95,
                'reason': f"The query asks about '{', '.join(distinctive_query_tokens[:3])}', which does not exist anywhere in the 'data/' knowledge base. The model correctly refused, exposing a data gap.",
                'suggested_fix': f"Add documentation covering this topic into 'data/' (e.g. update AboutUs.md with '{', '.join(distinctive_query_tokens[:3])}' details).",
                'details': {
                    'issue_type': 'missing_corpus_data',
                    'distinctive_query_tokens': distinctive_query_tokens
                }
            }

        # The corpus DOES have this knowledge. Check if the retriever actually included it.
        retrieved_matched_distinctive = tokens_overlap(distinctive_query_tokens, retrieved_texts)
        if len(retrieved_matched_distinctive) >= max(1, len(distinctive_query_tokens) * 0.5):
            return {
                'verdict': 'MODEL_ISSUE',
                'confidence': 0.88,
                'reason': f"Model falsely refused to answer even though relevant context for '{', '.join(retrieved_matched_distinctive)}' was provided in the prompt (False Refusal).",
                'suggested_fix': "Refine system prompt to instruct model not to refuse when context mentions the requested facts.",
                'details': {
                    'issue_type': 'false_refusal',
                    'retrieved_chunks_count': len(retrieved_chunks),
                    'matched_tokens': retrieved_matched_distinctive
                }
            }
        else:
            return {
                'verdict': 'RETRIEVAL_ISSUE',
                'confidence': 0.92,
                'reason': f"The knowledge exists in '{best_corpus_match['source']}' (section '{best_corpus_match['title']}'), but PageIndex failed to retrieve it (score was 0 or ranked outside top-3).",
                'suggested_fix': f"Enhance PageIndex: add keyword expansion or synonym matching so '{user_query}' retrieves section '{best_corpus_match['title']}'.",
                'details': {
                    'issue_type': 'retrieval_miss',
                    'missed_source': best_corpus_match['source'],
                    'missed_section': best_corpus_match['title'],
                    'corpus_match_score': best_corpus_match['score']
                }
            }

    # Case B: Model answered (did not refuse)
    if not has_retrieved_chunks or total_retrieval_score == 0:
        if corpus_has_answer:
            return {
                'verdict': 'RETRIEVAL_ISSUE',
                'confidence': 0.80,
                'reason': "Model answered, but retrieval returned 0 chunks. The model relied on outside pre-training knowledge instead of grounded context.",
                'suggested_fix': f"Improve retriever indexing so query '{user_query}' retrieves '{best_corpus_match['source']}'.",
                'details': {
                    'issue_type': 'retrieval_zero_hit',
                    'best_corpus_source': best_corpus_match['source']
                }
            }
        else:
            return {
                'verdict': 'MODEL_ISSUE',
                'confidence': 0.90,
                'reason': "Model hallucinated an answer using outside pre-training knowledge when 0 context was retrieved and data doesn't exist in data/ (Knowledge Rule violated).",
                'suggested_fix': "Instruct the LLM with higher penalty on hallucination and lower temperature.",
                'details': {
                    'issue_type': 'hallucination_outside_knowledge'
                }
            }

    # Case C: Context retrieved and model answered
    # Check if the model answer contains facts supported by retrieved chunks
    response_words = set(re.findall(r'[a-zA-Z0-9\u0900-\u097F]+', model_response.lower()))
    retrieved_words = set(re.findall(r'[a-zA-Z0-9\u0900-\u097F]+', retrieved_texts.lower()))
    
    # Overlap between model answer and retrieved context
    grounding_overlap = len(response_words.intersection(retrieved_words)) / max(1, len(response_words))

    # For cross-lingual answers (English context -> Nepali response), direct lexical overlap is low,
    # so we verify that chunks were retrieved and response is compliant
    is_cross_lingual = (lang_code == '0') and bool(re.search(r'[\u0900-\u097F]', model_response))
    if not is_cross_lingual and grounding_overlap < 0.15 and not corpus_has_distinctive:
        return {
            'verdict': 'MODEL_ISSUE',
            'confidence': 0.85,
            'reason': "Model response does not appear grounded in the retrieved context (potential hallucination).",
            'suggested_fix': "Instruct the model to cite or directly extract answers from the provided context.",
            'details': {
                'issue_type': 'unsupported_generation',
                'grounding_overlap': round(grounding_overlap, 2)
            }
        }

    return {
        'verdict': 'SUCCESS',
        'confidence': 0.90,
        'reason': "Context was properly retrieved from knowledge base, and model provided a grounded response complying with instructions.",
        'suggested_fix': "No fix needed. Pipeline operated as expected.",
        'details': {
            'retrieval_score': total_retrieval_score,
            'num_chunks': len(retrieved_chunks),
            'grounded': True
        }
    }


def diagnose_trace_llm_judge(trace: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Performs deep diagnosis using Ollama LLM-as-a-judge.
    """
    user_query = trace.get('user_query', '')
    lang_code = trace.get('language_code', '1')
    lang_name = 'English' if lang_code == '1' else 'Nepali (Devanagari)'
    retrieved_chunks = trace.get('retrieval_chunks') or []
    model_response = trace.get('model_response', '')

    context_str = "\n\n".join([
        f"Source: {c.get('source')} | Title: {c.get('title')}\n{c.get('content', '')[:400]}"
        for c in retrieved_chunks
    ]) if retrieved_chunks else "(No context retrieved)"

    corpus_matches = corpus_scanner.scan_for_query(user_query)
    corpus_hint = ""
    if corpus_matches and corpus_matches[0]['score'] >= 0.3:
        top_m = corpus_matches[0]
        corpus_hint = f"Note: File '{top_m['source']}' section '{top_m['title']}' contains: {top_m['content'][:300]}"

    judge_prompt = f"""You are an expert AI RAG Evaluation Auditor.
Analyze this interaction with Penny (an AI counsellor for Sunway College):

USER QUERY: "{user_query}"
EXPECTED LANGUAGE: {lang_name}
RETRIEVED CONTEXT FROM DATABASE:
{context_str}

WHOLE CORPUS SCAN HINT:
{corpus_hint if corpus_hint else "(Nothing found in whole corpus)"}

ACTUAL MODEL RESPONSE:
"{model_response}"

DIAGNOSTIC CRITERIA:
1. RETRIEVAL_ISSUE: The knowledge was in the corpus, but retriever failed to fetch it or fetched irrelevant chunks.
2. DATA_ISSUE: The information is completely missing from the Sunway College data corpus (knowledge gap).
3. MODEL_ISSUE: Retrieved context had the answer, but the model hallucinated, gave false refusal, or failed rules.
4. LANGUAGE_ISSUE: Model responded in the wrong language/script.
5. SUCCESS: Accurately answered based on retrieved context and followed language instructions.

Return ONLY a valid JSON object with these exact keys:
{{
  "verdict": "SUCCESS" | "RETRIEVAL_ISSUE" | "DATA_ISSUE" | "MODEL_ISSUE" | "LANGUAGE_ISSUE",
  "confidence": 0.0 to 1.0,
  "reason": "Clear explanation of what went wrong or why it succeeded",
  "suggested_fix": "Actionable advice to fix data, retriever, or prompt",
  "context_relevance_score": 1 to 5,
  "groundedness_score": 1 to 5
}}
"""

    try:
        res = requests.post(OLLAMA_URL, json={
            'model': model_name,
            'messages': [{'role': 'user', 'content': judge_prompt}],
            'think': False,
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 300}
        }, timeout=45)
        res.raise_for_status()
        raw = res.json().get('message', {}).get('content', '')
        
        # Clean markdown code blocks if returned
        clean_json = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
        clean_json = re.sub(r'```$', '', clean_json, flags=re.MULTILINE).strip()
        data = json.loads(clean_json)
        
        # Validate verdict
        valid_verdicts = {'SUCCESS', 'RETRIEVAL_ISSUE', 'DATA_ISSUE', 'MODEL_ISSUE', 'LANGUAGE_ISSUE'}
        verdict = data.get('verdict', '').upper()
        if verdict not in valid_verdicts:
            verdict = 'PENDING'
        data['verdict'] = verdict
        return data

    except Exception as e:
        print(f"⚠️ LLM Judge fallback to heuristic due to: {e}")
        return diagnose_trace_heuristic(trace)


def evaluate_trace(trace_id: str, use_llm_judge: bool = False) -> Dict[str, Any]:
    """
    Evaluates a specific trace by ID and writes the diagnosis back to the DB.
    """
    trace = get_trace(trace_id)
    if not trace:
        raise ValueError(f"Trace not found: {trace_id}")

    if use_llm_judge:
        diagnosis = diagnose_trace_llm_judge(trace)
    else:
        diagnosis = diagnose_trace_heuristic(trace)

    verdict = diagnosis.get('verdict', 'PENDING')
    reason = diagnosis.get('reason', '')
    suggested_fix = diagnosis.get('suggested_fix', '')
    details = diagnosis.get('details', {})

    update_trace_diagnosis(
        trace_id=trace_id,
        verdict=verdict,
        diagnosis_reason=reason,
        suggested_fix=suggested_fix,
        diagnosis_details=details
    )

    return diagnosis


def evaluate_all_pending(use_llm_judge: bool = False) -> Dict[str, int]:
    """
    Runs evaluation over all traces that currently have verdict 'PENDING'.
    """
    from .tracker import get_traces
    pending = get_traces(verdict_filter='PENDING', limit=500)
    count = 0
    breakdown = Counter()
    for t in pending:
        try:
            diag = evaluate_trace(t['id'], use_llm_judge=use_llm_judge)
            breakdown[diag.get('verdict', 'PENDING')] += 1
            count += 1
        except Exception as e:
            print(f"Error evaluating trace {t['id']}: {e}")

    return {
        'total_evaluated': count,
        'breakdown': dict(breakdown)
    }
