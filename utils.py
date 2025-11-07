import json, os
from typing import Dict, List, Tuple

def _norm_id(x) -> str:
    s = str(x)
    return f"{int(s):06d}" if s.isdigit() else s

def _take_topk_refs(obj, k: int) -> List[str]:
    if obj is None or k <= 0: return []
    if isinstance(obj, list):
        if not obj: return []
        x0 = obj[0]
        if isinstance(x0, str):                 return [_norm_id(x) for x in obj[:k]]
        if isinstance(x0, (list, tuple)):       # [["rid", score], ...]
            tmp = [(_norm_id(it[0]), float(it[1]) if len(it)>1 else 0.0) for it in obj]
            tmp.sort(key=lambda t: t[1], reverse=True)
            return [rid for rid,_ in tmp[:k]]
        if isinstance(x0, dict):                # [{"id"/"rid":..., "score":...}, ...]
            key = "id" if "id" in x0 else ("rid" if "rid" in x0 else None)
            if key:
                tmp = [(_norm_id(d[key]), float(d.get("score", 0.0))) for d in obj]
                tmp.sort(key=lambda t: t[1], reverse=True)
                return [rid for rid,_ in tmp[:k]]
        return [_norm_id(x) for x in obj[:k]]
    if isinstance(obj, dict):
        if "ids" in obj and isinstance(obj["ids"], list):
            return [_norm_id(x) for x in obj["ids"][:k]]
        if "list" in obj and isinstance(obj["list"], list):
            return [_norm_id(x) for x in obj["list"][:k]]
        # {rid: score}
        items = [(_norm_id(rid), float(sc) if isinstance(sc,(int,float)) else 0.0) for rid,sc in obj.items()]
        items.sort(key=lambda t: t[1], reverse=True)
        return [rid for rid,_ in items[:k]]
    return []

def build_ref_bank_and_mapping(meta_dir: str,
                               split_name: str,
                               rag_topk: int,
                               use_refs_json: bool = True
                               ) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Bank comes ONLY from data-split.json['reference'].
    references.json (if present & use_refs_json) is used ONLY to map per-SID top-k indices into that bank.
    Returns: (ref_bank_ids, sid_to_refidx)
    """
    # --- bank from split file
    split_path = os.path.join(meta_dir, "data-split.json")
    with open(split_path, "r") as f:
        splits = json.load(f)
    ref_raw = splits.get("reference", [])
    ref_ids = ref_raw[0] if (isinstance(ref_raw, list) and ref_raw and isinstance(ref_raw[0], list)) else ref_raw
    ref_bank_ids = sorted({_norm_id(x) for x in ref_ids})
    ref_id2idx = {rid: i for i, rid in enumerate(ref_bank_ids)}

    sid_to_refidx: Dict[str, List[int]] = {}
    if use_refs_json and rag_topk > 0:
        rjson = os.path.join(meta_dir, "references.json")
        if os.path.exists(rjson):
            with open(rjson, "r") as f:
                raw = json.load(f)
            ref_map = raw.get(split_name, raw) if isinstance(raw, dict) else raw
            for sid, lst in (ref_map.items() if isinstance(ref_map, dict) else []):
                ids = _take_topk_refs(lst, rag_topk)
                idxs = [ref_id2idx.get(rid, -1) for rid in ids]
                while len(idxs) < rag_topk: idxs.append(-1)
                sid_to_refidx[_norm_id(sid)] = idxs

    return ref_bank_ids, sid_to_refidx


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
