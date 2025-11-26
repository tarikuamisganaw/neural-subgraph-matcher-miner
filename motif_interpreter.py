# motif_interpreter_gemini_full.py
"""
Motif Interpreter — Gemini-powered interpretations + per-pattern Interpret button + compact summary + Q&A.

Requirements:
    pip install streamlit google-generativeai

Run:
    export GEMINI_API_KEY="your_key_here"
    streamlit run motif_interpreter_gemini_full.py

Notes:
 - The app reads ./results/patterns.json by default or accepts an uploaded JSON file.
 - Gemini key is taken from GEMINI_API_KEY environment variable if present.
 - Per-pattern Interpret buttons are cached in st.session_state["interpretations"] to avoid repeated API calls.
 - Q&A sits right below the compact summary (Option A).
"""

import os
import re
import json
import streamlit as st
from typing import List, Dict, Optional
from collections import Counter, defaultdict
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Gemini config (prefer env var) ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") 
USE_GEMINI = bool(GEMINI_KEY)
if USE_GEMINI:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)

# Default path for patterns
DEFAULT_PATH = "results/patterns.json"

# -------------------- Loading & normalization --------------------
def load_patterns_from_path(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None
    return normalize_input_data(data)

def load_patterns_from_upload(uploaded) -> Optional[List[Dict]]:
    try:
        raw = uploaded.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        st.error(f"Uploaded file not valid JSON: {e}")
        return None
    return normalize_input_data(data)

def normalize_input_data(data) -> List[Dict]:
    items = []
    if isinstance(data, dict):
        if "patterns" in data and isinstance(data["patterns"], list):
            items = data["patterns"]
        else:
            if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
                items = [data]
            else:
                cand = []
                for v in data.values():
                    if isinstance(v, dict) and ("nodes" in v or "edges" in v or "metadata" in v):
                        cand.append(v)
                items = cand
    elif isinstance(data, list):
        items = data
    else:
        items = []
    normalized = []
    for p in items:
        norm = _normalize_single_pattern(p)
        if norm:
            normalized.append(norm)
    return normalized

def _normalize_single_pattern(p: Dict) -> Optional[Dict]:
    if not isinstance(p, dict):
        return None
    nodes_raw = p.get("nodes", [])
    nodes = {}
    for n in nodes_raw:
        nid = n.get("id") or n.get("node_id") or n.get("address") or n.get("label")
        if not nid:
            continue
        label = n.get("label") or (nid[:12] + "...") if isinstance(nid, str) else str(nid)
        feat = n.get("feature") or n.get("features") or []
        try:
            feat_nums = [float(x) for x in feat] if isinstance(feat, (list, tuple)) else []
        except Exception:
            feat_nums = []
        nodes[nid] = {"label": label, "feature": feat_nums, "anchor": n.get("anchor", None)}
    edges_raw = p.get("edges", []) or p.get("links", []) or []
    edge_count = 0
    edge_types = []
    amounts = []
    for e in edges_raw:
        edge_count += 1
        et = e.get("type") or e.get("token") or e.get("label")
        if isinstance(et, str):
            edge_types.append(et)
        if "amount" in e:
            try:
                amounts.append(float(e["amount"]))
            except Exception:
                pass
    meta = p.get("metadata", {}) or {}
    size = meta.get("num_nodes") or (len(nodes) if nodes else None)
    if not size and nodes:
        size = len(nodes)
    if edge_count is None:
        edge_count = meta.get("num_edges") or len(edges_raw) or 0
    occurrences = p.get("occurrences") or meta.get("occurrences") or meta.get("count") or 1
    hub = None
    for nid, nd in nodes.items():
        if nd.get("anchor") == 1 or nd.get("anchor") is True:
            hub = nid
            break
    if hub is None:
        best = None
        best_val = -float("inf")
        for nid, nd in nodes.items():
            feats = nd.get("feature") or []
            if feats:
                try:
                    v = float(feats[0])
                    if v > best_val:
                        best_val = v
                        best = nid
                except Exception:
                    continue
        if best is not None:
            hub = best
    unique_edge_types = sorted(set(edge_types)) if edge_types else None
    normalized = {
        "size": int(size) if size is not None else None,
        "edge_count": int(edge_count) if edge_count is not None else None,
        "occurrences": int(occurrences) if isinstance(occurrences, (int, float)) else occurrences,
        "edge_types": unique_edge_types,
        "hub": hub,
        "nodes": nodes,
        "amounts": amounts,
        "raw_original": p
    }
    return normalized

# -------------------- Stats & context builder --------------------
def compute_stats(patterns: List[Dict]) -> Dict:
    stats = {}
    total = sum(1 for p in patterns if not p.get("parse_failed"))
    stats["total_patterns"] = total
    counts = Counter()
    token_counts = Counter()
    token_by_size = defaultdict(Counter)
    hub_counts = Counter()
    hub_strength = {}
    directed_count = 0
    total_volume_by_token = defaultdict(float)
    for p in patterns:
        if p.get("parse_failed"):
            continue
        size = p.get("size")
        counts[size] += 1
        for t in (p.get("edge_types") or []):
            token_counts[t] += 1
            token_by_size[size][t] += 1
        hub = p.get("hub")
        if hub:
            hub_counts[hub] += 1
            nd = (p.get("nodes") or {}).get(hub)
            if nd and nd.get("feature"):
                try:
                    v = float(nd["feature"][0])
                    prev = hub_strength.get(hub, -float("inf"))
                    if v > prev:
                        hub_strength[hub] = v
                except Exception:
                    pass
        meta = p.get("raw_original", {}).get("metadata", {}) or {}
        if meta.get("is_directed") is True:
            directed_count += 1
        for e in (p.get("raw_original", {}).get("edges") or []):
            tok = e.get("type") or e.get("label")
            amt = e.get("amount")
            if tok and amt:
                try:
                    total_volume_by_token[tok] += float(amt)
                except Exception:
                    pass
    stats["counts_by_size"] = dict(counts)
    stats["token_counts"] = dict(token_counts)
    stats["token_by_size"] = {k: dict(v) for k, v in token_by_size.items()}
    stats["top_hubs_by_occurrence"] = [{"hub": h, "count": c} for h, c in hub_counts.most_common(5)]
    stats["top_hubs_by_strength"] = [{"hub": h, "feature0": f} for h, f in sorted(hub_strength.items(), key=lambda x: -x[1])[:5]]
    stats["directed_count"] = directed_count
    stats["total_volume_by_token"] = dict(total_volume_by_token)
    return stats

def build_compact_context(patterns: List[Dict], limit: int = 20) -> str:
    lines = []
    for i, p in enumerate(patterns[:limit], 1):
        size = p.get("size", "?")
        edges = p.get("edge_count") or "?"
        occ = p.get("occurrences", 1)
        types = p.get("edge_types") or []
        types_s = ",".join(types) if types else "unlabeled"
        hub = p.get("hub") or "none"
        hub_short = (hub[:10] + "...") if isinstance(hub, str) and len(hub) > 10 else str(hub)
        lines.append(f"P{i}: {size}-node edges={edges} types=[{types_s}] hub={hub_short} ×{occ}")
    return "\n".join(lines)

# -------------------- Gemini helpers (interpret & QA) --------------------
def _extract_first_sentences(text: str, n: int = 4) -> str:
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n]).strip()

def interpret_with_gemini(motif: Dict, model_name: str = "gemini-2.5-flash", debug: bool = False) -> Dict:
    if not USE_GEMINI:
        return {"answer": "[Gemini not configured — set GEMINI_API_KEY]", "raw": None}

    size = motif.get("size", "?")
    edge_types = motif.get("edge_types") or []
    occ = motif.get("occurrences", 1)
    hub = motif.get("hub") or "none"
    hub_display = (hub[:8] + "...") if isinstance(hub, str) and len(hub) > 8 else hub
    et_display = ", ".join(edge_types) if edge_types else "unlabeled"

    prompt = (
        f"Graph motif: {size}-node; edge types: {et_display}; hub: {hub_display}; occurrences: {occ}.\n\n"
        "You are a concise graph-analysis expert. In 1-3 short, plain-English sentences:\n"
        " - describe the likely shape (chain, star/hub-and-spoke, cycle, clique, mixed),\n"
        " - ignore the number of nodes unless specifically asked about them.\n\n"
        " - explain what a hub node implies (if present), and\n"
        " - mention the edge types  and their implications \n"
        "Do NOT invent numbers. If there is insufficient detail, reply: 'Insufficient data.'\n\n"
        "Answer:"
    )

    raw_resp = None
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            raw_resp = response.text
            ans = _extract_first_sentences(raw_resp, 2)
            return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        out = getattr(response, "output", None)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                cont = first.get("content")
                if isinstance(cont, list) and cont and isinstance(cont[0], dict):
                    txt = cont[0].get("text")
                    raw_resp = txt
                    ans = _extract_first_sentences(txt, 2)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
                if "text" in first and isinstance(first["text"], str):
                    raw_resp = first["text"]
                    ans = _extract_first_sentences(raw_resp, 2)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        cands = getattr(response, "candidates", None)
        if isinstance(cands, list) and len(cands) > 0:
            cand0 = cands[0]
            cont = getattr(cand0, "content", None)
            if cont and getattr(cont, "parts", None):
                maybe = cont.parts[0].text
                raw_resp = maybe
                ans = _extract_first_sentences(maybe, 4)
                return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
            if isinstance(cand0, dict):
                cont = cand0.get("content")
                if isinstance(cont, dict) and cont.get("parts"):
                    txt = cont["parts"][0].get("text", "").strip()
                    raw_resp = txt
                    ans = _extract_first_sentences(txt, 4)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
                for v in cand0.values():
                    if isinstance(v, str) and v.strip():
                        raw_resp = v
                        ans = _extract_first_sentences(v, 4)
                        return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        s = str(response)
        raw_resp = s
        ans = _extract_first_sentences(s, 4)
        return {"answer": ans or "[Gemini returned no usable text]", "raw": raw_resp if debug else None}

    except Exception as e:
        return {"answer": f"[Gemini error: {type(e).__name__}: {e}]", "raw": None}

def answer_question_with_gemini(question: str, patterns: List[Dict], model_name: str = "gemini-2.5-flash", debug: bool = False) -> Dict:
    if not USE_GEMINI:
        return {"answer": "[Gemini not configured — set GEMINI_API_KEY]", "raw": None}
    if not patterns:
        return {"answer": "[No patterns loaded — load or upload patterns.json first]", "raw": None}

    context = build_compact_context(patterns, limit=20)
    prompt = prompt = (
    "You are a concise, domain-aware analyst that ONLY uses the CONTEXT below. "
    "The CONTEXT lists short structured pattern summaries (P1, P2, ...). "
    "Task: answer the user's QUESTION using ONLY facts present in CONTEXT. "
    "Describe the likely shape (chain, star/hub-and-spoke, cycle, clique, mixed), if asked genralizaion questions about patterns.\n\n"
    "ignore the number of nodes unless specifically asked about them.\n\n"
    "- If the question is definitional (e.g., 'what is a hub?'), answer *using only how it appears in these motifs*.\n\n"
    "If you refer to a specific pattern, prefix it with the pattern id (e.g. 'P3: ...'). "
    "Do NOT invent facts, numbers, or external data. If the context lacks evidence for a claim, reply exactly: 'Insufficient data.' "
    "When applicable, give one short domain insight (one sentence) that is phrased as a hypothesis and labeled 'Hypothesis:' — but only derive hypotheses from explicit context fields (edge types, hub, occurrences, volumes). "
    "Keep it natural: 2–5 clear, conversational sentences. Use phrases like:\n"
    "  • 'This looks like…'\n"
    "  • 'A likely explanation is…'\n"
    "  • 'Given the recurrence, it may serve as…'\n"
    "  • 'This resembles [domain-appropriate analogy] — for example…'\n\n"
    f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
)

    raw_resp = None
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            raw_resp = response.text
            ans = _extract_first_sentences(raw_resp, 2)
            return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        out = getattr(response, "output", None)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                cont = first.get("content")
                if isinstance(cont, list) and cont and isinstance(cont[0], dict):
                    txt = cont[0].get("text")
                    raw_resp = txt
                    ans = _extract_first_sentences(txt, 2)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
                if "text" in first and isinstance(first["text"], str):
                    raw_resp = first["text"]
                    ans = _extract_first_sentences(raw_resp, 2)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        cands = getattr(response, "candidates", None)
        if isinstance(cands, list) and cands:
            cand0 = cands[0]
            cont = getattr(cand0, "content", None)
            if cont and getattr(cont, "parts", None):
                txt = cont.parts[0].text
                raw_resp = txt
                ans = _extract_first_sentences(txt, 4)
                return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
            if isinstance(cand0, dict):
                cont = cand0.get("content")
                if isinstance(cont, dict) and cont.get("parts"):
                    txt = cont["parts"][0].get("text", "").strip()
                    raw_resp = txt
                    ans = _extract_first_sentences(txt, 4)
                    return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}
                for v in cand0.values():
                    if isinstance(v, str) and v.strip():
                        raw_resp = v
                        ans = _extract_first_sentences(v, 4)
                        return {"answer": ans or "[Gemini returned no text]", "raw": raw_resp if debug else None}

        s = str(response)
        raw_resp = s
        ans = _extract_first_sentences(s, 4)
        return {"answer": ans or "[Gemini returned no usable text]", "raw": raw_resp if debug else None}

    except Exception as e:
        return {"answer": f"[Q&A error: {type(e).__name__}: {e}]", "raw": None}

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Motif Interpreter", layout="wide")
st.title("🔍 Motif Interpreter")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Load patterns JSON")
    if os.path.exists(DEFAULT_PATH):
        st.success(f"Found {DEFAULT_PATH}")
        if st.button("Load ./results/patterns.json"):
            patterns = load_patterns_from_path(DEFAULT_PATH) or []
            st.session_state["patterns"] = patterns
            # reset interpretations cache when new patterns loaded
            st.session_state["interpretations"] = {}
            # compute stats & context automatically
            st.session_state["stats"] = compute_stats(patterns)
            st.session_state["context"] = build_compact_context(patterns, limit=20)
            st.success(f"Loaded {len(patterns)} patterns.")
    uploaded = st.file_uploader("Or upload a patterns.json file", type=["json"])
    if uploaded:
        patterns = load_patterns_from_upload(uploaded)
        if patterns is None:
            st.error("Uploaded JSON doesn't look like patterns. Expect list or {patterns:[...]} structure.")
        else:
            st.session_state["patterns"] = patterns
            st.session_state["interpretations"] = {}
            st.session_state["stats"] = compute_stats(patterns)
            st.session_state["context"] = build_compact_context(patterns, limit=20)
            st.success(f"Uploaded and loaded {len(patterns)} patterns.")
    st.markdown("---")
    # if USE_GEMINI:
    #     st.write("Gemini configured (GEMINI_API_KEY present).")
    # else:
    #     st.warning("Gemini not configured. Set GEMINI_API_KEY to enable Gemini-based interpretations.")

# If patterns loaded, show compact summary, Q&A, and pattern list
if "patterns" in st.session_state:
    patterns = st.session_state["patterns"]
    # ensure stats exist
    stats = st.session_state.get("stats") or compute_stats(patterns)

    st.subheader("Summary")
    counts_by_size = stats.get("counts_by_size", {})
    token_counts = stats.get("token_counts", {})
    top_hubs = stats.get("top_hubs_by_occurrence", [])
    directed_count = stats.get("directed_count", 0)

    summary_lines = []
    summary_lines.append(f"Total patterns: {stats.get('total_patterns', 0)}")
    if counts_by_size:
        summary_lines.append("Counts by size: " + ", ".join(f"{k}:{v}" for k, v in counts_by_size.items()))
    if token_counts:
        top_tokens = ", ".join(f"{k}({v})" for k, v in sorted(token_counts.items(), key=lambda x:-x[1])[:5])
        summary_lines.append("Top tokens: " + top_tokens)
    if top_hubs:
        hubs = ", ".join(f"{h['hub']}({h['count']})" for h in top_hubs[:5])
        summary_lines.append("Top hubs: " + hubs)
    if directed_count > 0:
        summary_lines.append(f"Directed patterns detected: {directed_count}")

    st.markdown("**Compact summary:**")
    st.code("\n".join(summary_lines))

    # ------------------ Q&A (Option A location: right below summary) ------------------
    st.subheader("Ask a question (Q&A)")
    question = st.text_input("Question (e.g.'How many 3-node patterns?', 'What tokens?')", "")
    if st.button("Ask Gemini"):
        res = answer_question_with_gemini(question, patterns, debug=False)
        st.markdown("**Answer:**")
        st.write(res["answer"])


    st.markdown("---")

    # ------------------ Patterns list with per-pattern Interpret button ------------------
    st.subheader("Patterns (click a pattern's Interpret button to see AI explanation)")
   

    # initialize interpretations cache if missing
    if "interpretations" not in st.session_state:
        st.session_state["interpretations"] = {}

    # render pattern list (expanders keep things compact; large lists are scrollable in the page)
    for i, p in enumerate(patterns, 1):
        display = {
            "size": p.get("size"),
            "edge_count": p.get("edge_count"),
            "occurrences": p.get("occurrences"),
            "edge_types": p.get("edge_types"),
            "hub": (p.get("hub")[:10] + "...") if isinstance(p.get("hub"), str) and len(p.get("hub")) > 10 else p.get("hub")
        }
        with st.expander(f"Pattern {i} — size={display['size']} — hub={display['hub']}"):
            st.json(display)
            col_a, col_b = st.columns([1, 3])
            with col_a:
                if st.button(f"Interpret pattern {i}", key=f"interp_btn_{i}"):
                    res = interpret_with_gemini(p, debug=False)
                    st.session_state["interpretations"][i] = res
            with col_b:
                cached = st.session_state["interpretations"].get(i)
                if cached:
                    st.markdown(cached["answer"])
                    if cached.get("raw"):
                        st.markdown("**Raw Gemini response (debug):**")
                        st.code(cached["raw"])
                else:
                    st.info(f"No interpretation yet. Click 'Interpret pattern {i}' to get one.")

else:
    st.info("No patterns loaded yet. Load ./results/patterns.json or upload a file.")

st.markdown("---")
st.caption("Gemini-powered interpretation & Q&A. Use the per-pattern Interpret button and debug toggles to inspect raw Gemini responses for prompt tuning.")
