"""
Data processing and statistics functions
"""
from typing import List, Dict
from collections import Counter, defaultdict
from loaders import get_graph_context
from logging_setup import logger



def compute_stats(patterns: List[Dict]) -> Dict:
    """Compute statistics from patterns"""
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
    total_original_instances = 0
    total_duplicates_removed = 0
    rank_distribution = Counter()
    pattern_keys = []
    
    for p in patterns:
        if p.get("parse_failed"):
            continue
        
        size = p.get("size")
        counts[size] += 1
        total_original_instances += p.get("original_count", 1)
        total_duplicates_removed += p.get("duplicates_removed", 0)
        rank_distribution[p.get("rank", 0)] += 1
        pattern_keys.append(p.get("pattern_key", ""))
        
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
    stats["total_original_instances"] = total_original_instances
    stats["total_duplicates_removed"] = total_duplicates_removed
    stats["deduplication_rate"] = total_duplicates_removed / max(total_original_instances, 1)
    stats["rank_distribution"] = dict(rank_distribution)
    stats["unique_pattern_keys"] = len(set(pattern_keys))
    
    return stats


def build_compact_context(patterns: List[Dict], limit: int = 20) -> str:
    """Build compact context string for Gemini prompts"""
    lines = []
    
    # Add graph context if available
    ctx = get_graph_context()
    if ctx:
        lines.append(
            f"GRAPH CONTEXT: {ctx.get('total_nodes', 0):,} nodes, "
            f"{ctx.get('total_edges', 0):,} edges, "
            f"{ctx.get('sampling_trials', 0):,} trials, "
            f"{ctx.get('neighborhoods_sampled', 0):,} neighborhoods sampled"
        )
        lines.append("")
    
    for i, p in enumerate(patterns[:limit], 1):
        size = p.get("size", "?")
        edges = p.get("edge_count") or "?"
        orig_count = p.get("original_count", 1)
        freq_score = p.get("frequency_score", 0)
        dup_removed = p.get("duplicates_removed", 0)
        pattern_key = p.get("pattern_key", "")
        types = p.get("edge_types") or []
        types_s = ",".join(types) if types else "unlabeled"
        hub = p.get("hub") or "none"
        hub_short = (hub[:10] + "...") if isinstance(hub, str) and len(hub) > 10 else str(hub)
        
        context_line = (
            f"P{i}: {pattern_key} {size}-node edges={edges} "
            f"types=[{types_s}] hub={hub_short} ×{orig_count} "
            f"freq={freq_score:.3f} dups={dup_removed}"
        )
        lines.append(context_line)
    
    return "\n".join(lines)


def validate_and_report(parsed_obj, title="Parse report"):
    """
    Diagnostic helper for reporting parsing results
    """
    from loaders import get_graph_context
    
    logger.info("validate_and_report: %s", title)
    ctx = get_graph_context()
    logger.info("validate_and_report: graph_context is %s", "set" if ctx else "None")
    
    try:
        if ctx and isinstance(ctx, dict):
            keys = list(ctx.keys())
            logger.info("validate_and_report: graph_context keys (first 20): %s", keys[:20])
    except Exception:
        logger.exception("validate_and_report: unexpected error")