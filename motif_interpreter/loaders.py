"""
Data loading and normalization functions
"""
import os
import json
from typing import List, Dict, Optional
from logging_setup import logger

# Global graph context
graph_context = None


def _try_parse_ndjson_text(text: str) -> Optional[List]:
    """
    Tolerant NDJSON parser
    """
    objs = []
    for i, raw_ln in enumerate(text.splitlines()):
        ln = raw_ln.strip()
        if not ln:
            continue
        if ln.endswith(","):
            ln = ln[:-1].rstrip()
        if not (ln.startswith("{") or ln.startswith("[")):
            logger.debug("Skipping non-json-looking line %d: %r", i, ln[:80])
            continue
        try:
            objs.append(json.loads(ln))
        except Exception as e:
            logger.debug("Failed to parse line %d as JSON: %s", i, e)
            continue
    return objs if objs else None


def _extract_graph_context_from_obj(obj: dict) -> Optional[Dict]:
    """Try multiple ways to find graph_context in an object."""
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "graph_context":
        return obj.get("data") or obj.get("context") or {}
    if "graph_context" in obj and isinstance(obj["graph_context"], dict):
        return obj["graph_context"]
    ctx = obj.get("context") or obj.get("data")
    if isinstance(ctx, dict) and ("total_nodes" in ctx or "sampling_trials" in ctx or "total_edges" in ctx):
        return ctx
    return None


def load_patterns_from_file(path: str) -> Optional[List[Dict]]:
    """
    Load patterns from a file path
    """
    global graph_context
    graph_context = None

    logger.info("Loading patterns from path: %s", path)
    
    if not os.path.exists(path):
        logger.warning("Path not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        logger.exception("Failed to read file %s", path)
        return None

    # Try parse JSON then NDJSON
    data = None
    try:
        data = json.loads(raw_text)
        logger.info("Parsed file as JSON (top-level type=%s)", type(data).__name__)
    except Exception as e_json:
        logger.warning("json.loads failed: %s", e_json)
        nd = _try_parse_ndjson_text(raw_text)
        if nd is not None:
            data = nd
            logger.info("Parsed as NDJSON with %d objects", len(nd))
        else:
            logger.exception("JSON and NDJSON parsing both failed for %s", path)
            return None

    return _process_parsed_data(data)


def load_patterns_from_upload(uploaded_file) -> Optional[List[Dict]]:
    """
    Load patterns from uploaded file
    """
    global graph_context
    graph_context = None

    try:
        raw = uploaded_file.read()
        raw_text = raw.decode("utf-8")
    except Exception as e:
        logger.exception("Uploaded JSON decode failed")
        return None

    # Try normal JSON parsing, then NDJSON
    try:
        data = json.loads(raw_text)
    except Exception as e:
        logger.warning("json.loads failed (%s). Trying NDJSON parse...", e)
        nd = _try_parse_ndjson_text(raw_text)
        if nd is not None:
            data = nd
            logger.info("Parsed as NDJSON with %d lines", len(nd))
        else:
            logger.exception("Uploaded JSON parse failed")
            return None

    return _process_parsed_data(data)


def _process_parsed_data(data) -> Optional[List[Dict]]:
    """
    Process parsed data to extract patterns and graph context
    """
    global graph_context
    
    if isinstance(data, dict):
        ctx_candidate = _extract_graph_context_from_obj(data)
        if ctx_candidate:
            graph_context = ctx_candidate
            logger.info("Extracted graph_context from top-level dict")
        
        if isinstance(data.get("patterns"), list):
            items = data["patterns"]
            return _normalize_input_data(items)
        return _normalize_input_data(data)

    if isinstance(data, list):
        kept = []
        found_ctx = False
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                ctx = _extract_graph_context_from_obj(item)
                if ctx is not None and not found_ctx:
                    graph_context = ctx
                    found_ctx = True
                    logger.info("Found graph_context in array at index %d", idx)
                    continue
            kept.append(item)
        if found_ctx:
            data = kept
        return _normalize_input_data(data)

    logger.warning("Unexpected top-level parsed type: %s", type(data))
    return _normalize_input_data(data)


def _normalize_input_data(data) -> List[Dict]:
    """
    Normalize input data into list of pattern dicts
    """
    items = []
    logger.debug("normalize_input_data: top-level type=%s", type(data))

    if isinstance(data, dict):
        if "patterns" in data and isinstance(data["patterns"], list):
            items = data["patterns"]
        elif isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
            items = [data]
        else:
            cand = []
            for k, v in data.items():
                if isinstance(v, dict) and ("nodes" in v or "edges" in v or "metadata" in v):
                    cand.append(v)
            items = cand

    elif isinstance(data, list):
        items = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "graph_context":
                global graph_context
                if graph_context is None:
                    graph_context = item.get("data") or item.get("context") or {}
                    logger.info("Discovered graph_context in array (late extraction).")
                continue
            items.append(item)

    else:
        logger.warning("Unrecognized JSON top-level type: %s", type(data))

    normalized = []
    for i, p in enumerate(items):
        try:
            norm = _normalize_single_pattern(p)
            if norm:
                normalized.append(norm)
        except Exception as e:
            logger.exception("Exception while normalizing item %d: %s", i, e)

    logger.info("Normalized %d patterns", len(normalized))
    return normalized


def _normalize_single_pattern(p: Dict) -> Optional[Dict]:
    """Normalize a single pattern dictionary"""
    if not isinstance(p, dict):
        return None

    # Extract nodes
    nodes = _extract_nodes(p.get("nodes", []))
    
    # Extract edges
    edges_info = _extract_edges_info(p.get("edges", []) or p.get("links", []))
    
    # Extract metadata
    meta = _extract_metadata(p)
    
    # Build normalized pattern
    return _build_normalized_pattern(nodes, edges_info, meta, p)


def _extract_nodes(nodes_raw):
    """Extract and normalize nodes"""
    nodes = {}
    
    if isinstance(nodes_raw, dict):
        for nid, nd in nodes_raw.items():
            label = None
            feat = []
            anchor = None
            if isinstance(nd, dict):
                label = nd.get("label") or (str(nid)[:12] + "...")
                feat = nd.get("feature") or nd.get("features") or []
                anchor = nd.get("anchor", None)
            else:
                label = str(nid)
            try:
                feat_nums = [float(x) for x in feat] if isinstance(feat, (list, tuple)) else []
            except Exception:
                feat_nums = []
            nodes[nid] = {"label": label, "feature": feat_nums, "anchor": anchor}
    else:
        for n in nodes_raw or []:
            if not isinstance(n, dict):
                continue
            nid = n.get("id") or n.get("node_id") or n.get("address") or n.get("label")
            if not nid:
                continue
            label = n.get("label") or (nid[:12] + "...") if isinstance(nid, str) else str(nid)
            feat = n.get("feature") or n.get("features") or []
            try:
                feat_nums = [float(x) for x in feat] if isinstance(feat, (list, tuple)) else []
            except Exception:
                feat_nums = []
            anchor = n.get("anchor", None)
            nodes[nid] = {"label": label, "feature": feat_nums, "anchor": anchor}
    
    return nodes


def _extract_edges_info(edges_raw):
    """Extract edge information"""
    edge_count = 0
    edge_types = []
    amounts = []
    
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        edge_count += 1
        et = e.get("type") or e.get("token") or e.get("label")
        if isinstance(et, str):
            edge_types.append(et)
        if "amount" in e:
            try:
                amounts.append(float(e["amount"]))
            except Exception:
                pass
    
    return {
        "count": edge_count,
        "types": edge_types,
        "amounts": amounts,
        "unique_types": sorted(set(edge_types)) if edge_types else None
    }


def _extract_metadata(p):
    """Extract metadata from pattern"""
    meta = p.get("metadata", {}) or p.get("meta", {}) or {}
    
    if not meta:
        if isinstance(p.get("raw_original"), dict):
            meta_cand = p["raw_original"].get("metadata")
            if isinstance(meta_cand, dict):
                meta = meta_cand
    
    if not meta:
        found_keys = {}
        for key in ("original_count", "frequency_score", "duplicates_removed", 
                   "rank", "pattern_key", "size", "num_nodes", "num_edges"):
            if key in p:
                found_keys[key] = p[key]
        if found_keys:
            meta = found_keys
    
    return meta


def _build_normalized_pattern(nodes, edges_info, meta, original_p):
    """Build normalized pattern dictionary"""
    pattern_key = meta.get("pattern_key", "") or original_p.get("pattern_key", "")
    size = meta.get("size") or meta.get("num_nodes") or (len(nodes) if nodes else None)
    edge_count_meta = meta.get("num_edges") or edges_info["count"]
    
    # Find hub
    hub = _find_hub(nodes)
    
    # Convert values with safe fallbacks
    try:
        size_i = int(size) if size is not None else None
    except Exception:
        size_i = None
    
    try:
        edge_count_i = int(edge_count_meta) if edge_count_meta is not None else None
    except Exception:
        edge_count_i = None
    
    original_count = meta.get("original_count", original_p.get("original_count", 1))
    frequency_score = meta.get("frequency_score", original_p.get("frequency_score", 0))
    duplicates_removed = meta.get("duplicates_removed", original_p.get("duplicates_removed", 0))
    rank = meta.get("rank", original_p.get("rank", 0))
    
    # Safe conversions
    try:
        original_count_i = int(original_count) if isinstance(original_count, (int, float, str)) and str(original_count).isdigit() else original_count
    except Exception:
        original_count_i = original_count
    
    try:
        frequency_score_f = float(frequency_score) if frequency_score is not None else 0.0
    except Exception:
        frequency_score_f = frequency_score
    
    try:
        duplicates_removed_i = int(duplicates_removed) if isinstance(duplicates_removed, (int, float, str)) and str(duplicates_removed).isdigit() else duplicates_removed
    except Exception:
        duplicates_removed_i = duplicates_removed
    
    try:
        rank_i = int(rank) if isinstance(rank, (int, float, str)) and str(rank).isdigit() else rank
    except Exception:
        rank_i = rank
    
    return {
        "size": size_i,
        "edge_count": edge_count_i,
        "original_count": original_count_i,
        "pattern_key": pattern_key,
        "rank": rank_i,
        "frequency_score": frequency_score_f,
        "duplicates_removed": duplicates_removed_i,
        "edge_types": edges_info["unique_types"],
        "hub": hub,
        "nodes": nodes,
        "amounts": edges_info["amounts"],
        "raw_original": original_p
    }


def _find_hub(nodes):
    """Find hub node in pattern"""
    hub = None
    
    # First check for anchor nodes
    for nid, nd in nodes.items():
        if nd.get("anchor") == 1 or nd.get("anchor") is True:
            return nid
    
    # Otherwise find node with highest first feature value
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
    
    return best


def get_graph_context():
    """Get the current graph context"""
    global graph_context
    return graph_context


def set_graph_context(context):
    """Set the graph context"""
    global graph_context
    graph_context = context