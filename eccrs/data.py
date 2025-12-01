# eccrs/data.py
# Utilities to load the LP dataset and build bitset views.
# Supports both a(j) and a(j,k) feature atoms.

import re
from typing import Dict, List, Tuple, Set, Optional

Row = Dict[int, int]  # internal feature id -> {0,1}

# Matches:
# val(ROW, a(J), V).
# val(ROW, a(J,K), V).
LP_LINE_RE = re.compile(
    r"val\(\s*(\d+)\s*,\s*a\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\)\s*,\s*(0|1)\s*\)\s*\.\s*"
)


class Dataset:
    def __init__(
        self,
        rows: List[Row],
        labels: List[int],
        features: List[int],
        label_attr: int,
        feat_names: Optional[Dict[int, str]] = None,
    ):
        self.rows = rows  # list of dicts: internal_feature_id -> 0/1
        self.labels = labels  # list of 0/1
        self.n = len(rows)  # number of rows
        self.features = features  # internal feature ids (exclude label attr and any ignored)
        self.label_attr = label_attr
        self.feat_names: Dict[int, str] = feat_names or {j: f"a({j})" for j in features}

        # Build fast bitsets for each signed literal (attr=j, val=v)
        self.bitsets: Dict[Tuple[int, int], int] = {}
        all_bits = (1 << self.n) - 1
        for j in self.features:
            for v in (0, 1):
                bits = 0
                for i, row in enumerate(self.rows):
                    if row.get(j) == v:
                        bits |= (1 << i)
                self.bitsets[(j, v)] = bits

        # Label bitsets
        self.Y_pos = 0
        self.Y_neg = 0
        for i, y in enumerate(self.labels):
            if y == 1:
                self.Y_pos |= (1 << i)
            else:
                self.Y_neg |= (1 << i)
        self.ALL = (1 << self.n) - 1

        # Cache literals true on each row for reuse
        self.row_literals = [
            {(j, v) for j, v in row.items()}
            for row in self.rows
        ]


def load_lp(path: str, label_attr: int = 10, ignore_attrs: Set[int] = None) -> Dataset:
    """
    Reads lines like:
      val(RowID, a(j), v).
      val(RowID, a(j,k), v).
    # categorical one-hot features

    label_attr: the 'j' such that a(j) is the label (1/0). Label is assumed unary.
    ignore_attrs: set of 'j' to skip entirely (applies to both a(j) and all a(j, *)).
    """
    if ignore_attrs is None:
        ignore_attrs = set()

    # First pass: collect raw entries and track max unary attribute id
    raw_entries: List[Tuple[int, int, Optional[int], int]] = []  # (rid, j, k_or_None, v)
    max_unary_j = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LP_LINE_RE.match(line)
            if not m:
                continue
            rid = int(m.group(1))
            j = int(m.group(2))
            k = int(m.group(3)) if m.group(3) is not None else None
            v = int(m.group(4))
            raw_entries.append((rid, j, k, v))
            if k is None:
                if j > max_unary_j:
                    max_unary_j = j

    # Map composite (j,k) to unique internal feature ids beyond max_unary_j
    next_id = max_unary_j + 1
    jk_to_id: Dict[Tuple[int, int], int] = {}
    id_to_name: Dict[int, str] = {}

    def get_feat_id(j: int, k: Optional[int]) -> Optional[int]:
        # Skip ignored attributes completely
        if j in ignore_attrs:
            return None
        if k is None:
            # plain unary attribute id stays as-is
            return j
        key = (j, k)
        fid = jk_to_id.get(key)
        if fid is None:
            fid = next_id_map[0]
            jk_to_id[key] = fid
            id_to_name[fid] = f"a({j},{k})"
            next_id_map[0] += 1
        return fid

    next_id_map = [next_id]  # small trick to allow closure mutation

    # Build rows grouped by rid
    triples_by_row: Dict[int, Dict[int, int]] = {}
    label_by_row: Dict[int, int] = {}

    for rid, j, k, v in raw_entries:
        if k is None and j == label_attr:
            # label (must be unary)
            label_by_row[rid] = v
            continue
        fid = get_feat_id(j, k)
        if fid is None:
            continue  # ignored attribute
        row = triples_by_row.setdefault(rid, {})
        row[fid] = v
        # For unary features, remember pretty name only if not set
        if k is None and fid not in id_to_name:
            id_to_name[fid] = f"a({j})"

    # Normalize rows (only those with labels present)
    rows: List[Row] = []
    labels: List[int] = []
    feature_ids: Set[int] = set()

    for rid in sorted(triples_by_row.keys()):
        if rid not in label_by_row:
            continue
        y = int(label_by_row[rid])
        feat_row = triples_by_row[rid]
        rows.append(feat_row)
        labels.append(y)
        feature_ids.update(feat_row.keys())

    features_sorted = sorted(feature_ids)
    feat_names = {fid: id_to_name.get(fid, f"a({fid})") for fid in features_sorted}

    return Dataset(
        rows=rows,
        labels=labels,
        features=features_sorted,
        label_attr=label_attr,
        feat_names=feat_names,
    )


# ===== Fast-path caches for kNN and optional vectorized distances =====


def _ensure_vecs(ds: "Dataset"):
    """Dense 0/1 vectors per row in feature order (built once and cached)."""
    vecs = getattr(ds, "_vecs", None)
    if vecs is None:
        feats = ds.features
        ds._vecs = [[row.get(j, 0) for j in feats] for row in ds.rows]
    return ds._vecs


def _ensure_bitpack(ds: "Dataset"):
    """Pack each row into a Python int (bitset) for ultra-fast unweighted Hamming."""
    words = getattr(ds, "_bitpack", None)
    if words is not None:
        return words
    feats = ds.features
    F = len(feats)
    W = (F + 63) // 64  # not strictly used, kept for clarity
    packed = []
    for r in ds.rows:
        x = 0
        for t, j in enumerate(feats):
            if r.get(j, 0):
                x |= (1 << t)
        packed.append(x)
    ds._bitpack = (packed, F, W)
    return ds._bitpack


def _maybe_np_matrix(ds: "Dataset"):
    """Optional NumPy boolean matrix for fast weighted Hamming.

    distance(row) = (train_bool ^ row_bool) @ weights

    Created only if NumPy is available; otherwise returns (None, None).
    """
    np = None
    try:
        import numpy as _np  # type: ignore

        np = _np
    except Exception:
        return None, None

    mat = getattr(ds, "_np_bool", None)
    if mat is None:
        feats = ds.features
        mat = np.zeros((ds.n, len(feats)), dtype=bool)
        for i, r in enumerate(ds.rows):
            for c, j in enumerate(feats):
                mat[i, c] = bool(r.get(j, 0))
        ds._np_bool = mat
    return np, ds._np_bool
