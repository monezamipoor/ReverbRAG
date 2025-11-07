def _take_topk_refs(obj, k: int):
    """Return a list[str] of top-k reference ids from various formats."""
    if obj is None:
        return []
    # list[str]
    if isinstance(obj, list):
        if not obj:
            return []
        # list[str]
        if isinstance(obj[0], str):
            return obj[:k]
        # list[[rid, score]]
        if isinstance(obj[0], (list, tuple)):
            tmp = []
            for it in obj:
                rid = str(it[0])
                sc  = float(it[1]) if (len(it) > 1 and isinstance(it[1], (int, float))) else 0.0
                tmp.append((rid, sc))
            tmp.sort(key=lambda t: t[1], reverse=True)
            return [rid for rid, _ in tmp[:k]]
        # list[dict] like {"id"/"rid": "...", "score": ...}
        if isinstance(obj[0], dict):
            cand_key = "id" if "id" in obj[0] else ("rid" if "rid" in obj[0] else None)
            if cand_key is not None:
                tmp = [(str(d[cand_key]), float(d.get("score", 0.0))) for d in obj]
                tmp.sort(key=lambda t: t[1], reverse=True)
                return [rid for rid, _ in tmp[:k]]
            # fallback
            return [str(x) for x in obj[:k]]
        # generic fallback
        return [str(x) for x in obj[:k]]
    # dict cases
    if isinstance(obj, dict):
        # {"ids":[...]} or {"list":[...]}
        if "ids" in obj and isinstance(obj["ids"], list):
            return [str(x) for x in obj["ids"][:k]]
        if "list" in obj and isinstance(obj["list"], list):
            return [str(x) for x in obj["list"][:k]]
        # {rid: score}
        try:
            items = [(str(rid), float(sc) if isinstance(sc, (int, float)) else 0.0)
                     for rid, sc in obj.items()]
            items.sort(key=lambda t: t[1], reverse=True)
            return [rid for rid, _ in items[:k]]
        except Exception:
            return []
    return []
