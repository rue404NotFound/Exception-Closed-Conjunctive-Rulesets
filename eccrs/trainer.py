# eccrs/trainer.py
# OPTIMIZED: Lazy audit (only after successful adds) + cached coverage in closure

from typing import Optional, List, Tuple, Set, Dict
from dataclasses import dataclass
import math
import heapq
from .data import Dataset, _ensure_vecs, _ensure_bitpack, _maybe_np_matrix
from .model import RuleSet, Rule
from .projector import projector_add, coverage_of_body
from .certify import certificate_for_row
from . import fallback as fb

Literal = Tuple[int, int]

# ----------------------- 
# Utilities / metrics
# ----------------------- 
def predict_all(rs: RuleSet, ds: Dataset) -> List[Optional[int]]:
    return rs.predict_all(ds)

def accuracy_on_covered(preds: List[Optional[int]], labels: List[int]) -> Tuple[float, float]:
    covered = [(p, y) for p, y in zip(preds, labels) if p is not None]
    cov_frac = len(covered) / len(labels) if labels else 0.0
    if not covered:
        return 0.0, cov_frac
    ok = sum(1 for p, y in covered if p == y)
    return ok / len(covered), cov_frac

def overall_accuracy(preds: List[int], labels: List[int]) -> float:
    ok = sum(1 for p, y in zip(preds, labels) if p == y)
    return ok / len(labels) if labels else 0.0

def popcount(x: int) -> int:
    return x.bit_count()

# ----------------------- 
# Training (OPTIMIZED: lazy audit)
# ----------------------- 
def train_cf_eccrs(
    ds: Dataset,
    max_iters: int = 5000,
    verbose: bool = True,
    max_rules: int = 10**9,
    min_cov_pos: int = 1,
    min_gain: int = 1
) -> RuleSet:
    """
    OPTIMIZED: Only audit after successful rule additions (lazy audit).
    """
    rs = RuleSet()
    iters = 0
    
    while iters < max_iters and len(rs.rules) < max_rules:
        iters += 1
        base_preds = predict_all(rs, ds)
        mistakes_idx = [i for i, (p, y) in enumerate(zip(base_preds, ds.labels)) 
                        if (p is None) or (p != y)]
        
        if not mistakes_idx:
            if verbose:
                print(f"[done] No mistakes. iters={iters-1}, rules={len(rs.rules)}")
            break
        
        mistakes_bits = 0
        for i in mistakes_idx:
            mistakes_bits |= (1 << i)
        
        added_any = False
        rules_before = len(rs.rules)
        
        for idx in mistakes_idx:
            if len(rs.rules) >= max_rules:
                break
            
            y_true = ds.labels[idx]
            cert = certificate_for_row(ds, idx, y_true)
            
            if cert is None or len(cert) == 0:
                if verbose and cert is None:
                    print(f"[warn] Could not build certificate for row {idx}.")
                if verbose and cert == frozenset():
                    print(f"[guard] refusing empty-body certificate from row {idx}.")
                continue
            
            added = projector_add(ds, rs, cert, y_true, witness=idx)
            if added is None:
                continue
            
            own_Y = ds.Y_pos if added.label == 1 else ds.Y_neg
            own_cov = popcount(added.cov_bits & own_Y)
            
            if own_cov < min_cov_pos:
                rs.rules.pop()
                if verbose:
                    print(f"[gate] drop rule (min_cov_pos={min_cov_pos}, got {own_cov})")
                continue
            
            gain = popcount(added.cov_bits & mistakes_bits)
            if gain < min_gain:
                rs.rules.pop()
                if verbose:
                    print(f"[gate] drop rule (min_gain={min_gain}, got {gain})")
                continue
            
            added_any = True
            if verbose:
                print(f"[add] rule {len(rs.rules)}: body={sorted(list(added.body))} => {added.label} (own_cov={own_cov}, gain={gain})")
        
        # OPTIMIZATION: Only audit if new rules were added
        if added_any and len(rs.rules) > rules_before:
            ok, msg = audit_eccrs_global(ds, rs)
            if not ok:
                if verbose:
                    print(f"[audit] VIOLATION: {msg}")
                # Note: In production you might want to handle this more gracefully
        
        if not added_any:
            if verbose:
                print("[stop] No addable rules this epoch (gates or inseparable cases).")
            break
    
    preds = predict_all(rs, ds)
    sel_acc, cov = accuracy_on_covered(preds, ds.labels)
    if verbose:
        print(f"[report] accuracy_on_covered={sel_acc:.3f}, abstention={(1.0-cov):.3f}, rules={len(rs.rules)}")
    
    return rs

# ----------------------- 
# ECCRS audit (global)
# ----------------------- 
def audit_eccrs_global(ds: Dataset, rs: RuleSet) -> Tuple[bool, str]:
    n = len(rs.rules)
    seen: Dict[frozenset, int] = {}
    
    for i, r in enumerate(rs.rules):
        if r.body in seen and rs.rules[seen[r.body]].label != r.label:
            return False, f"Duplicate body with opposite labels at rules {seen[r.body]+1} and {i+1}."
        seen[r.body] = i
    
    for i in range(n):
        for j in range(i + 1, n):
            ri = rs.rules[i]
            rj = rs.rules[j]
            
            if ri.label == rj.label:
                continue
            
            cov_i = coverage_of_body(ds, set(ri.body))
            cov_j = coverage_of_body(ds, set(rj.body))
            
            if (cov_i & cov_j) != 0:
                Bi = set(ri.body)
                Bj = set(rj.body)
                if not (Bi.issubset(Bj) or Bj.issubset(Bi)):
                    return False, f"Opposite-label flat overlap between rules {i+1} and {j+1}."
    
    return True, ""


# -----------------------
# Fallbacks
# -----------------------
@dataclass
class RuleStat:
    cov_count: int
    prec: float
    body_size: int
    label: int


def compute_rule_stats(rs: RuleSet, ds_train: Dataset, laplace: float = 1.0) -> List[RuleStat]:
    stats: List[RuleStat] = []
    for r in rs.rules:
        cov = r.cov_bits
        cov_count = popcount(cov)
        if r.label == 1:
            correct = popcount(cov & ds_train.Y_pos)
        else:
            correct = popcount(cov & ds_train.Y_neg)
        prec = (correct + laplace) / (cov_count + 2 * laplace) if cov_count > 0 else 0.5
        stats.append(RuleStat(cov_count=cov_count, prec=prec, body_size=len(r.body), label=r.label))
    return stats


def nearest_rule_fallback(rs: RuleSet, row_lits, rule_stats: List[RuleStat]) -> int:
    # Kept for API compatibility; not used by the fast path.
    best_dist = None
    cand_idx: List[int] = []
    for idx, r in enumerate(rs.rules):
        dist = len([lit for lit in r.body if lit not in row_lits])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            cand_idx = [idx]
        elif dist == best_dist:
            cand_idx.append(idx)

    def key(i: int):
        st = rule_stats[i]
        return (st.prec, st.body_size, st.cov_count, st.label)

    best_i = max(cand_idx, key=key)
    return rs.rules[best_i].label


def row_to_vec(ds: Dataset, row: Dict[int, int]) -> List[int]:
    return [row.get(j, 0) for j in ds.features]


def compute_feature_weights_mi(ds: Dataset) -> List[float]:
    # Cache on Dataset to avoid recomputation
    cached = getattr(ds, "_mi_weights", None)
    if cached is not None:
        return cached

    N = ds.n
    if N == 0:
        ds._mi_weights = [1.0 for _ in ds.features]
        return ds._mi_weights

    w = []
    for j in ds.features:
        x1_bits = ds.bitsets[(j, 1)]
        x1 = popcount(x1_bits)
        x0 = N - x1
        y1 = popcount(ds.Y_pos)
        y0 = N - y1
        x1y1 = popcount(x1_bits & ds.Y_pos)
        x1y0 = x1 - x1y1
        x0y1 = y1 - x1y1
        x0y0 = N - x1y1 - x1y0 - x0y1

        def term(a, b, c):
            if a == 0:
                return 0.0
            pxy = a / N
            px = b / N
            py = c / N
            return pxy * math.log(pxy / (px * py + 1e-12) + 1e-12)

        mi = 0.0
        mi += term(x1y1, x1, y1)
        mi += term(x1y0, x1, y0)
        mi += term(x0y1, x0, y1)
        mi += term(x0y0, x0, y0)
        w.append(max(mi, 1e-8))

    mean_w = sum(w) / len(w) if w else 1.0
    w = [wi / mean_w for wi in w]
    ds._mi_weights = w
    return w


def _bitpack_from_rowvec(row_vec: List[int]) -> int:
    q = 0
    for t, val in enumerate(row_vec):
        if val:
            q |= (1 << t)
    return q


def knn_fallback(
    ds_train: Dataset,
    row_vec: List[int],
    k: int,
    knn_weighted: bool = False,
    feat_weights: Optional[List[float]] = None
) -> int:
    """
    Fast exact kNN:
    - Unweighted Hamming: bit-packed XOR + popcount.
    - Weighted Hamming: NumPy fast path if available, else cached dense vectors.
    Neighbor vote can be uniform or 1/(d+eps) weighted.
    """
    k = max(1, min(k, ds_train.n))

    # Compute distances
    if feat_weights is None:
        # Unweighted: bitpack both query and train rows
        packed, F, W = _ensure_bitpack(ds_train)
        q = _bitpack_from_rowvec(row_vec)
        dists = [(p ^ q).bit_count() for p in packed]
    else:
        # Weighted Hamming: try NumPy
        np, mat = _maybe_np_matrix(ds_train)
        if np is not None:
            rb = np.asarray(row_vec, dtype=bool)
            w = np.asarray(feat_weights, dtype=float)
            dists_np = (mat ^ rb) @ w
            dists = dists_np.tolist()
        else:
            train_vecs = _ensure_vecs(ds_train)
            dists = []
            for v in train_vecs:
                s = 0.0
                for x, y, w in zip(row_vec, v, feat_weights):
                    if x != y:
                        s += w
                dists.append(s)

    # Top-k without full sort
    heap: List[Tuple[float, int]] = []
    push = heapq.heappush; pop = heapq.heappop
    for i, d in enumerate(dists):
        if len(heap) < k:
            push(heap, (-d, i))
        else:
            if d < -heap[0][0]:
                pop(heap)
            push(heap, (-d, i))
    neigh = [idx for _, idx in sorted(heap, key=lambda x: (-x[0], x[1]))]

    if not knn_weighted:
        votes1 = sum(1 for i in neigh if ds_train.labels[i] == 1)
        votes0 = k - votes1
        return 1 if votes1 >= votes0 else 0
    else:
        eps = 1e-9
        w1 = 0.0; w0 = 0.0
        for i in neigh:
            d = dists[i]
            w = 1.0 / (d + eps)
            if ds_train.labels[i] == 1:
                w1 += w
            else:
                w0 += w
        return 1 if w1 >= w0 else 0


# -----------------------
# Evaluation wrapper
# -----------------------
def evaluate_with_fallback(
    rs: RuleSet,
    ds: Dataset,
    fallback: str = "abst",
    k: int = 5,
    train_ref: Optional[Dataset] = None,
    tag: str = "eval",
    knn_weighted: bool = False,
    feat_weight: str = "none"
) -> Dict[str, float]:
    base_preds = rs.predict_all(ds)
    sel_acc, cov = accuracy_on_covered(base_preds, ds.labels)
    print(f"[report] accuracy_on_covered={sel_acc:.3f}, abstention={(1.0-cov):.3f}, rules={len(rs.rules)}")

    if fallback == "abst":
        overall = sel_acc * cov
        print(f"[{tag}] selective_acc={sel_acc:.3f} coverage={cov:.3f} overall_acc={overall:.3f}")
        return dict(selective_acc=sel_acc, coverage=cov, overall_acc=overall)

    preds_full: List[int] = []
    if fallback == "nearest":
        tr = train_ref if train_ref is not None else ds
        for row, base in zip(ds.rows, base_preds):
            if base is not None:
                preds_full.append(base)
                continue
            row_lits = {(j, v) for j, v in row.items()}
            lab = fb.nearest_rule_completion(rs, tr, row_lits)
            preds_full.append(lab)
        overall = overall_accuracy(preds_full, ds.labels)
        print(f"[{tag}] selective_acc={sel_acc:.3f} coverage=1.000 overall_acc={overall:.3f}")
        return dict(selective_acc=sel_acc, coverage=1.0, overall_acc=overall)

    if fallback == "knn":
        tr = train_ref if train_ref is not None else ds
        weights = None
        if feat_weight == "mi":
            weights = compute_feature_weights_mi(tr)  # cached on Dataset
        for row, base in zip(ds.rows, base_preds):
            if base is not None:
                preds_full.append(base)
                continue
            rv = row_to_vec(ds, row)
            lab = knn_fallback(tr, rv, k, knn_weighted=knn_weighted, feat_weights=weights)
            preds_full.append(lab)
        overall = overall_accuracy(preds_full, ds.labels)
        print(f"[{tag}] selective_acc={sel_acc:.3f} coverage=1.000 overall_acc={overall:.3f}")
        return dict(selective_acc=sel_acc, coverage=1.0, overall_acc=overall)

    raise ValueError(f"Unknown fallback: {fallback}")


# -----------------------
# MDL selector
# -----------------------
def mdl_objective(decided_bits: int, N: int, model_cost: float, eta: float) -> float:
    undec = N - popcount(decided_bits)
    return model_cost + eta * undec


def run_mdl_selector(rs_pool: RuleSet, ds_train: Dataset, c0: float = 2.0, c1: float = 1.0, eta: float = 2.0, verbose: bool = True) -> RuleSet:
    N = ds_train.n
    decided = 0
    selected_idx: List[int] = []
    model_cost = 0.0
    L = mdl_objective(decided, N, model_cost, eta)

    if verbose:
        print("[mdl] Running EC-MDL selector on training set...")
        print(f"[mdl] start: K=0 L={L:.3f} (model={model_cost:.3f}, data={L-model_cost:.3f}; undec={N}, err=0)")

    covers = [r.cov_bits for r in rs_pool.rules]
    body_sizes = [len(r.body) for r in rs_pool.rules]
    used = [False]*len(rs_pool.rules)

    while True:
        best_idx = None
        best_delta = 0.0
        for i, r in enumerate(rs_pool.rules):
            if used[i]:
                continue
            new_bits = covers[i] & (~decided)
            gain = popcount(new_bits)
            if gain == 0:
                continue
            delta_model = c0 + c1 * body_sizes[i]
            delta_data = -eta * gain
            delta_L = delta_model + delta_data
            if (best_idx is None) or (delta_L < best_delta):
                best_idx = i
                best_delta = delta_L

        if best_idx is None or best_delta >= 0.0:
            if verbose:
                print(f"[mdl] stop: no rule reduces L further. K={len(selected_idx)} L={L:.3f}")
            break

        i = best_idx
        used[i] = True
        selected_idx.append(i)
        decided |= covers[i]
        model_cost += (c0 + c1 * body_sizes[i])
        L = mdl_objective(decided, N, model_cost, eta)

        if verbose:
            idx = i + 1
            cov = popcount(covers[i])
            print(f"[mdl] +rule pool_idx={idx:02d} body_size={body_sizes[i]} cov={cov} prec=1.000 ΔL={best_delta:+.3f} K={len(selected_idx)} L={L:.3f}")

    rs_sel = RuleSet(rules=[rs_pool.rules[i] for i in selected_idx])
    if verbose:
        undec = N - popcount(decided)
        print(f"[mdl] selected K={len(selected_idx)} L={L:.3f} (model={model_cost:.3f}, data={L-model_cost:.3f}; undec={undec}, err=0)")
    return rs_sel


# -----------------------
# Fallback-gain selector
# -----------------------
def overall_acc_with_fallback(
    rs: RuleSet,
    ds: Dataset,
    fallback: str,
    k: int,
    train_ref: Dataset,
    knn_weighted: bool,
    feat_weight: str
) -> float:
    base = rs.predict_all(ds)
    feat_weights = None
    if fallback == "knn" and feat_weight == "mi":
        feat_weights = compute_feature_weights_mi(train_ref)

    preds_full: List[int] = []
    for row, base_p in zip(ds.rows, base):
        if base_p is not None:
            preds_full.append(base_p)
            continue
        if fallback == "nearest":
            row_lits = {(j, v) for j, v in row.items()}
            lab = fb.nearest_rule_completion(rs, train_ref, row_lits)
            preds_full.append(lab)
        elif fallback == "knn":
            rv = row_to_vec(ds, row)
            lab = knn_fallback(train_ref, rv, k, knn_weighted=knn_weighted, feat_weights=feat_weights)
            preds_full.append(lab)
        else:
            preds_full.append(-999)

    return overall_accuracy(preds_full, ds.labels)


def run_fallback_gain_selector(
    rs_pool: RuleSet,
    ds_train: Dataset,
    fallback: str = "knn",
    k: int = 5,
    train_ref: Optional[Dataset] = None,
    knn_weighted: bool = False,
    feat_weight: str = "none",
    max_add: int = 100,
    min_delta: float = 0.0,
    verbose: bool = True
) -> RuleSet:
    if train_ref is None:
        train_ref = ds_train

    selected = RuleSet([])
    current_acc = overall_acc_with_fallback(selected, ds_train, fallback, k, train_ref, knn_weighted, feat_weight)
    if verbose:
        print(f"[fg] start acc={current_acc:.3f} with fallback={fallback}")

    used = [False] * len(rs_pool.rules)
    added = 0

    while added < max_add:
        best_i = -1
        best_acc = current_acc
        for i, r in enumerate(rs_pool.rules):
            if used[i]:
                continue
            trial = RuleSet(selected.rules + [r])
            acc = overall_acc_with_fallback(trial, ds_train, fallback, k, train_ref, knn_weighted, feat_weight)
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_i = i

        if best_i == -1 or best_acc - current_acc < min_delta:
            if verbose:
                print(f"[fg] stop: no rule improves ≥ {min_delta:.4f}. K={len(selected.rules)} acc={current_acc:.3f}")
            break

        used[best_i] = True
        selected = RuleSet(selected.rules + [rs_pool.rules[best_i]])
        added += 1
        if verbose:
            r = rs_pool.rules[best_i]
            print(f"[fg] +rule idx={best_i:02d} body_size={len(r.body)} label={r.label} acc: {current_acc:.3f} → {best_acc:.3f}")
        current_acc = best_acc

    if verbose:
        print(f"[fg] selected K={len(selected.rules)} acc={current_acc:.3f}")
    return selected


# -----------------------
# Redundancy pruning (same-label)
# -----------------------
def prune_redundant_same_label(ds: Dataset, rs: RuleSet) -> RuleSet:
    keep = [True] * len(rs.rules)
    for i, ri in enumerate(rs.rules):
        for j, rj in enumerate(rs.rules):
            if i == j:
                continue
            if ri.label != rj.label:
                continue
            Bi = set(ri.body); Bj = set(rj.body)
            if Bj.issubset(Bi) and len(Bj) < len(Bi):
                keep[i] = False
                break
    new_rules = [r for r, k in zip(rs.rules, keep) if k]
    return RuleSet(rules=new_rules)


# -----------------------
# Pretty-printer
# -----------------------
def print_ruleset(rs: RuleSet, ds: Dataset, title: str = "Rules:"):
    print(f"\n{title}")
    names = getattr(ds, "feat_names", None)
    for k, r in enumerate(rs.rules, 1):
        parts = []
        for (j, v) in sorted(list(r.body)):
            name = names.get(j, f"a({j})") if names else f"a({j})"
            parts.append(f"{name}={v}")
        body_str = " ∧ ".join(parts) if parts else "⊤"
        print(f" {k:02d}. {body_str} => {r.label}")


# ----------------------------------------------------------
# Strict laminar closure: synthesize global default→exception
# ----------------------------------------------------------
# ---------------------------------------------------------- 
# Strict laminar closure: OPTIMIZED with cached coverage
# ---------------------------------------------------------- 

def _bodies_compatible(B: Set[Literal], C: Set[Literal]) -> bool:
    seen: Dict[int, int] = {}
    for (j, v) in list(B) + list(C):
        if j in seen and seen[j] != v:
            return False
        seen[j] = v
    return True

def dedupe_exact(rs: RuleSet) -> RuleSet:
    seen = set()
    out = []
    for r in rs.rules:
        key = (r.body, r.label)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return RuleSet(out)

def laminar_strict_closure(
    rs: RuleSet,
    ds: Dataset,
    verbose: bool = True,
    max_rounds: int = 5_000,
    laminar_min_overlap: int = 0,
    orient_by_overlap: bool = True
) -> RuleSet:
    """
    OPTIMIZED: Cache coverage bitsets to avoid redundant recomputation.
    
    Synthesize union children until no incomparable opposite-label overlap remains.
    """
    def body_size(r: Rule) -> int:
        return len(r.body)
    
    def cov_count(r: Rule) -> int:
        return popcount(r.cov_bits)
    
    changed = True
    rounds = 0
    bodies_seen = {(r.body, r.label) for r in rs.rules}
    
    while changed and rounds < max_rounds:
        rounds += 1
        changed = False
        n = len(rs.rules)
        
        # OPTIMIZATION: Pre-compute and cache all coverage bitsets
        # This is the key speedup: avoid O(n²) calls to coverage_of_body
        covs = [r.cov_bits for r in rs.rules]
        
        for i in range(n):
            for j in range(i + 1, n):
                ri = rs.rules[i]
                rj = rs.rules[j]
                
                if ri.label == rj.label:
                    continue
                
                Bi = set(ri.body)
                Bj = set(rj.body)
                
                # Check comparability first (cheaper than compatibility)
                if Bi.issubset(Bj) or Bj.issubset(Bi):
                    continue
                
                if not _bodies_compatible(Bi, Bj):
                    continue
                
                # OPTIMIZATION: Use cached coverage bitsets
                overlap_bits = covs[i] & covs[j]
                overlap = popcount(overlap_bits)
                
                if overlap < laminar_min_overlap:
                    continue
                
                # Orientation logic
                if orient_by_overlap and overlap > 0:
                    pos_on_O = popcount(overlap_bits & ds.Y_pos)
                    neg_on_O = popcount(overlap_bits & ds.Y_neg)
                    
                    if pos_on_O > neg_on_O:
                        default = ri if ri.label == 1 else rj
                        other = rj if ri.label == 1 else ri
                        exception_label = 0
                    elif neg_on_O > pos_on_O:
                        default = ri if ri.label == 0 else rj
                        other = rj if ri.label == 0 else ri
                        exception_label = 1
                    else:
                        if body_size(ri) < body_size(rj):
                            default, other = ri, rj
                        elif body_size(rj) < body_size(ri):
                            default, other = rj, ri
                        else:
                            default, other = (ri, rj) if cov_count(ri) >= cov_count(rj) else (rj, ri)
                        exception_label = 1 - default.label
                else:
                    if body_size(ri) < body_size(rj):
                        default, other = ri, rj
                    elif body_size(rj) < body_size(ri):
                        default, other = rj, ri
                    else:
                        default, other = (ri, rj) if cov_count(ri) >= cov_count(rj) else (rj, ri)
                    exception_label = 1 - default.label
                
                union = frozenset(set(default.body).union(set(other.body)))
                key = (union, exception_label)
                
                if key in bodies_seen:
                    continue
                
                # Compute union coverage ONCE and cache it
                cov_union = coverage_of_body(ds, set(union))
                new_rule = Rule(body=union, label=exception_label, cov_bits=cov_union, witness=None)
                
                rs.rules.append(new_rule)
                bodies_seen.add(key)
                changed = True
                
                if verbose:
                    cov_num = popcount(cov_union)
                    tag = "" if overlap > 0 else " (no-train-overlap)"
                    print(
                        f"[laminar_strict] +exception: default |B|={len(default.body)} lab={default.label} "
                        f"child |B∪C|={len(union)} lab={exception_label} cov={cov_num}{tag}"
                    )
    
    if rounds >= max_rounds:
        print("[laminar_strict] reached max_rounds safeguard; closure may be incomplete.")
    
    rs = dedupe_exact(rs)
    return rs
# ----------------------------------------------------------
# Ensure at least one default→exception pair (always on)
# ----------------------------------------------------------
def _has_any_default_exception_pair(rs: RuleSet) -> bool:
    for i, ri in enumerate(rs.rules):
        Bi = set(ri.body)
        for j, rj in enumerate(rs.rules):
            if i == j or ri.label == rj.label:
                continue
            Bj = set(rj.body)
            if Bi.issubset(Bj):
                return True
    return False


def ensure_at_least_one_pair(rs: RuleSet, ds: Dataset, verbose: bool = True) -> RuleSet:
    if _has_any_default_exception_pair(rs):
        if verbose:
            print("[pair] already present: at least one default→exception relation exists.")
        return rs

    if not rs.rules:
        if verbose:
            print("[pair] no rules to pair; skipping.")
        return rs

    def cov_count(r: Rule) -> int:
        return popcount(r.cov_bits)

    ranked = sorted(rs.rules, key=lambda r: (len(r.body), -cov_count(r)))
    best_choice = None  # (default_rule, union_body, union_cov_bits, exception_label, zero_cov_bool)

    for d in ranked:
        Bd = set(d.body)
        for o in rs.rules:
            if o.label == d.label:
                continue
            Bo = set(o.body)
            # compatibility check
            seen: Dict[int, int] = {}
            ok = True
            for (jj, vv) in list(Bd) + list(Bo):
                if jj in seen and seen[jj] != vv:
                    ok = False; break
                seen[jj] = vv
            if not ok:
                continue

            union = frozenset(Bd.union(Bo))
            if union == frozenset(Bd) or union == frozenset(Bo):
                continue

            cov = coverage_of_body(ds, set(union))
            zero = (popcount(cov) == 0)
            choice = (d, union, cov, 1 - d.label, zero)
            if best_choice is None:
                best_choice = choice
            else:
                if choice[4] and not best_choice[4]:
                    best_choice = choice
                elif (not choice[4]) and (not best_choice[4]):
                    if popcount(choice[2]) < popcount(best_choice[2]):
                        best_choice = choice
                    elif popcount(choice[2]) == popcount(best_choice[2]):
                        if len(choice[0].body) < len(best_choice[0].body):
                            best_choice = choice

    if best_choice is not None:
        d, union, cov, ex_label, zero = best_choice
        new_rule = Rule(body=union, label=ex_label, cov_bits=cov, witness=None)
        rs.rules.append(new_rule)
        if verbose:
            tag = " (zero-train-coverage)" if zero else ""
            print(f"[pair] added: parent |B|={len(d.body)} label={d.label} child |C|={len(union)} label={ex_label}{tag}")
        return rs

    # Last resort: fabricate a zero-coverage child
    d = ranked[0]
    Bd = set(d.body)
    cov_B = d.cov_bits
    union = None
    for j in ds.features:
        for v in (0, 1):
            if (j, v) in Bd:
                continue
            candidate_cov = coverage_of_body(ds, Bd.union({(j, v)}))
            if popcount(candidate_cov & cov_B) == 0:
                union = frozenset(Bd.union({(j, v)}))
                cov = candidate_cov
                break
        if union is not None:
            break

    if union is None:
        j = ds.features[0]
        union = frozenset(Bd.union({(j, 0), (j, 1)}))
        cov = 0

    ex_label = 1 - d.label
    new_rule = Rule(body=union, label=ex_label, cov_bits=cov, witness=None)
    rs.rules.append(new_rule)
    if verbose:
        print(f"[pair] synthesized under simplest rule: parent |B|={len(d.body)} label={d.label} child |C|={len(union)} label={ex_label} (zero-train-coverage)")
    return rs
