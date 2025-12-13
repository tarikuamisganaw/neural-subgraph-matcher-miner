# motif_interpreter/main.py
import streamlit as st
import os

# FIXED: Direct imports from same directory
try:
    # Try importing directly
    from loaders import load_patterns_from_file, load_patterns_from_upload
    from processors import compute_stats, build_compact_context, validate_and_report
    from gemini_client import interpret_pattern, answer_question
    from config import DEFAULT_PATH
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all files are in the same directory.")
    raise

# Set up page config
st.set_page_config(page_title="Motif Interpreter", layout="wide")
st.title("🔍 Motif Interpreter")

# Create two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Load patterns JSON")
    
    # Show whether default file exists
    exists = os.path.exists(DEFAULT_PATH)
    if exists:
        st.success(f"Found {DEFAULT_PATH}")
    else:
        st.warning(f"{DEFAULT_PATH} not found (check working directory where you run streamlit).")
    
    # Load from default path button
    if st.button(f"Load {DEFAULT_PATH}"):
        patterns = load_patterns_from_file(DEFAULT_PATH) or []
        
        if patterns:
            st.session_state["patterns"] = patterns
            st.session_state["interpretations"] = {}
            st.session_state["stats"] = compute_stats(patterns)
            st.session_state["context"] = build_compact_context(patterns, limit=20)
            st.success(f"Loaded {len(patterns)} patterns.")
        else:
            st.error("No patterns loaded. Check terminal/log output for parsing errors.")
        
        validate_and_report(patterns, title=f"After loading {DEFAULT_PATH}")
    
    # File uploader
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
        
        validate_and_report(patterns, title="After upload")
    
    st.markdown("---")

# Main content area
with col2:
    if "patterns" in st.session_state:
        patterns = st.session_state["patterns"]
        stats = st.session_state.get("stats") or compute_stats(patterns)
        
        # Render summary
        st.subheader("Summary")
        counts_by_size = stats.get("counts_by_size", {})
        token_counts = stats.get("token_counts", {})
        top_hubs = stats.get("top_hubs_by_occurrence", [])
        directed_count = stats.get("directed_count", 0)
        total_original = stats.get("total_original_instances", 0)
        total_dups = stats.get("total_duplicates_removed", 0)
        dedup_rate = stats.get("deduplication_rate", 0)
        unique_patterns = stats.get("unique_pattern_keys", 0)
        
        summary_lines = []
        summary_lines.append(f"Total patterns: {stats.get('total_patterns', 0)}")
        summary_lines.append(f"Unique pattern types: {unique_patterns}")
        summary_lines.append(f"Total instances found: {total_original}")
        summary_lines.append(f"Duplicates removed: {total_dups} ({dedup_rate:.1%})")
        
        if counts_by_size:
            summary_lines.append("Counts by size: " + ", ".join(f"{k}:{v}" for k, v in counts_by_size.items()))
        
        if token_counts:
            top_tokens = ", ".join(f"{k}({v})" for k, v in sorted(token_counts.items(), key=lambda x: -x[1])[:5])
            summary_lines.append("Top tokens: " + top_tokens)
        
        if top_hubs:
            hubs = ", ".join(f"{h['hub']}({h['count']})" for h in top_hubs[:5])
            summary_lines.append("Top hubs: " + hubs)
        
        if directed_count > 0:
            summary_lines.append(f"Directed patterns detected: {directed_count}")
        
        st.markdown("**Compact summary:**")
        st.code("\n".join(summary_lines))
        
        # Render Q&A section
        st.subheader("Ask a question (Q&A)")
        question = st.text_input("Question (e.g.'How many 3-node patterns?', 'What tokens?')", "")
        
        if st.button("Ask Gemini"):
            res = answer_question(question, patterns, debug=False)
            st.markdown("**Answer:**")
            st.write(res["answer"])
        
        st.markdown("---")
        
        # Render patterns list
        st.subheader("Patterns (click a pattern's Interpret button to see AI explanation)")
        
        if "interpretations" not in st.session_state:
            st.session_state["interpretations"] = {}
        
        for i, p in enumerate(patterns, 1):
            display = {
                "size": p.get("size"),
                "edge_count": p.get("edge_count"),
                "original_count": p.get("original_count"),
                "frequency_score": p.get("frequency_score"),
                "duplicates_removed": p.get("duplicates_removed"),
                "rank": p.get("rank"),
                "edge_types": p.get("edge_types"),
                "hub": (p.get("hub")[:10] + "...") if isinstance(p.get("hub"), str) and len(p.get("hub")) > 10 else p.get("hub")
            }
            
            with st.expander(f"Pattern {i} — size={display['size']} — hub={display['hub']}"):
                st.json(display)
                col_a, col_b = st.columns([1, 3])
                
                with col_a:
                    if st.button(f"Interpret pattern {i}", key=f"interp_btn_{i}"):
                        res = interpret_pattern(p, debug=False)
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
        st.info("No patterns loaded yet. Load patterns_all_instances.json or upload a file.")

st.markdown("---")
st.caption("Gemini-powered interpretation & Q&A. Terminal logs show metadata extraction debug info.")