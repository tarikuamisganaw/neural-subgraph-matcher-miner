"""
Gemini API client and prompt construction
"""
import re
from typing import Dict, List, Optional
import google.generativeai as genai
from config import GEMINI_KEY, USE_GEMINI, GEMINI_MODEL
from processors import build_compact_context
from loaders import get_graph_context
from logging_setup import logger


def _extract_first_sentences(text: str, n: int = 4) -> str:
    """Extract first n sentences from text"""
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n]).strip()


def interpret_pattern(motif: Dict, model_name: str = GEMINI_MODEL, debug: bool = False) -> Dict:
    """Interpret a single pattern with Gemini"""
    if not USE_GEMINI:
        return {"answer": "[Gemini not configured — set GEMINI_API_KEY]", "raw": None}
    
    # Prepare pattern information
    size = motif.get("size", "?")
    edge_types = motif.get("edge_types") or []
    orig_count = motif.get("original_count", 1)
    freq_score = motif.get("frequency_score", 0)
    dup_removed = motif.get("duplicates_removed", 0)
    pattern_key = motif.get("pattern_key", "")
    hub = motif.get("hub") or "none"
    hub_display = (hub[:8] + "...") if isinstance(hub, str) and len(hub) > 8 else hub
    et_display = ", ".join(edge_types) if edge_types else "unlabeled"
    
    # Construct prompt
    prompt_parts = []
    prompt_parts.append(
        f"Graph pattern {pattern_key}: edge types: {et_display}; "
        f"hub: {hub_display}; original_count: {orig_count}; frequency_score: {float(freq_score):.3f}; "
        f"duplicates_removed: {dup_removed}.\n\n"
    )
    
    # Add graph context
    ctx = get_graph_context()
    if ctx and isinstance(ctx, dict):
        total_nodes = ctx.get("total_nodes", "unknown")
        total_edges = ctx.get("total_edges", "unknown")
        sampling_trials = ctx.get("sampling_trials", "unknown")
        
        try:
            total_nodes_s = f"{int(total_nodes):,}"
        except Exception:
            total_nodes_s = str(total_nodes)
        
        try:
            total_edges_s = f"{int(total_edges):,}"
        except Exception:
            total_edges_s = str(total_edges)
        
        try:
            sampling_trials_s = f"{int(sampling_trials):,}"
        except Exception:
            sampling_trials_s = str(sampling_trials)
        
        prompt_parts.append(
            f"GRAPH CONTEXT: Discovered in graph with {total_nodes_s} nodes and {total_edges_s} edges "
            f"using {sampling_trials_s} sampling trials.\n\n"
        )
    else:
        prompt_parts.append("GRAPH CONTEXT: Not available.\n\n")
    
    # Add instructions
    prompt_parts.append(
        "CRITICAL: You MUST use the GRAPH CONTEXT SUMMARY above when interpreting this single pattern and explain it according to that sampled context.\n"
        "IMPORTANT:Do NOT mention the number of nodes and size. Do NOT invent numbers."
        "These patterns were discovered through SAMPLING, not exhaustive graph analysis.\n"
        "- original_count: Raw discoveries before deduplication (includes duplicates from sampling)\n"
        "- frequency_score: Discovery rate relative to sampling trials (e.g., 0.06 = found in 6% of trials)\n"
        "- duplicates_removed: How many identical instances were found during sampling\n"
        "- rank: Relative importance among discovered patterns of same size\n\n"
        "You are a concise graph-analysis expert. In 3-6 short, plain-English sentences:\n"
        " - describe the likely shape (chain, star/hub-and-spoke, cycle, clique, mixed),\n"
        " - explain what the hub node implies (if present),\n"
        " - mention the edge types and their implications,\n"
        " - interpret the sampling metrics (frequency_score indicates structural significance, duplicates suggest concentrated vs widespread patterns)according to that sampled context,and mention it.\n\n"
        "If there is insufficient detail, reply exactly: 'Insufficient data.'\n\n"
        "Answer:"
    )
    
    prompt = "".join(prompt_parts)
    logger.debug("interpret_with_gemini: final prompt preview: %s", prompt[:1000])
    
    return _call_gemini_api(prompt, model_name, debug)


def answer_question(question: str, patterns: List[Dict], model_name: str = GEMINI_MODEL, debug: bool = False) -> Dict:
    """Answer questions about patterns using Gemini"""
    if not USE_GEMINI:
        return {"answer": "[Gemini not configured — set GEMINI_API_KEY]", "raw": None}
    if not patterns:
        return {"answer": "[No patterns loaded]", "raw": None}
    
    context = build_compact_context(patterns, limit=20)
    prompt = (
        "You are a concise, domain-aware analyst that ONLY uses the CONTEXT below. "
        "The CONTEXT lists short structured pattern summaries (P1, P2, ...). "
        "IMPORTANT:Do NOT mention the number of nodes and edges and size,especially  don't mention it in result generalization.\n"
        "CRITICAL: You MUST use the GRAPH CONTEXT SUMMARY above when interpreting this single pattern and explain it acoording to that sampled context."
        "CRITICAL SAMPLING CONTEXT: All patterns were discovered through SAMPLING of graphs, "
        "NOT exhaustive analysis of entire target graphs. Metrics reflect only sampled regions, "
        "not complete graph structures.\n\n"
        "CONTEXT METRICS EXPLANATION:\n"
        "- original_count: Raw discoveries in samples (includes duplicates from sampling)\n"
        "- frequency_score: Discovery rate in sampling trials (qualitative: high/medium/low)\n"
        "- duplicates_removed: How many identical instances found during sampling\n"
        "- rank: Relative importance among discovered patterns of same size\n\n"
        "Task: answer the user's QUESTION using ONLY facts present in CONTEXT. "
        "Describe likely shapes (chain, star/hub-and-spoke, cycle, clique, mixed) for pattern questions.\n\n"
        "IMPORTANT: Always emphasize that insights are based on SAMPLED data, not complete graphs. "
        "Use qualitative descriptions (common/rare, widespread/concentrated) rather than exact numbers.\n\n"
        "If you refer to a specific pattern, prefix it with the pattern id (e.g. 'P3: ...'). "
        "Do NOT invent facts, numbers, or external data. If the context lacks evidence for a claim, reply exactly: 'Insufficient data.' "
        "When applicable, give one short domain insight (one sentence) that is phrased as a hypothesis and labeled 'Hypothesis:' — but only derive hypotheses from explicit context fields (edge types, hub, qualitative frequency). "
        "Must be: 3–6 clear, conversational sentences. Use phrases like:\n"
        "  • 'Based on the samples, this looks like…'\n"
        "  • 'A likely explanation from the sampled data is…'\n"
        "  • 'Given the recurrence in samples, it may serve as…'\n"
        "  • 'This resembles [domain-appropriate analogy] — for example…'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    
    return _call_gemini_api(prompt, model_name, debug)


def _call_gemini_api(prompt: str, model_name: str, debug: bool = False) -> Dict:
    """Make API call to Gemini"""
    raw_resp = None
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            raw_resp = response.text
            ans = _extract_first_sentences(raw_resp, 2)
            return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
        
        # Fallback parsing logic
        # ... (keep your existing fallback logic here)
        
        s = str(response)
        raw_resp = s
        ans = _extract_first_sentences(s, 4)
        return {"answer": ans or "[Gemini returned no usable text]", "raw": raw_resp if debug else None}
        
    except Exception as e:
        logger.exception("Gemini API call failed")
        return {"answer": f"[Gemini error: {type(e).__name__}: {e}]", "raw": None}