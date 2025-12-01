# eccrs/selector.py
# EC-MDL selector for ECCRS rule sets (purely symbolic).

from typing import List, Tuple, Optional
import math
from .data import Dataset
from .model import Rule, RuleSet

def popcount(x: int) -> int:
    return x.bit_count()

def rule_body_size(r: Rule) -> int:
    return len(r.body)

def rule_cov_count(r: Rule) -> int:
    return popcount(r.cov_bits)

def rule_precision(r: Rule, ds: Dataset) -> float:
    cov = r.cov_bits
    tot = popcount(cov)
    if tot == 0:
        return 1.0
    if r.label == 1:
        correct = popcount(cov & ds.Y_pos)
    else:
        correct = popcount(cov & ds.Y_neg)
    return correct / tot

def predict_row_core(rs: RuleSet, row) -> Optional[int]:
    """ECCRS core prediction without fallback (row is a dict)."""
    return rs.predict_row(row)

def predict_all_core(rs: RuleSet, ds: Dataset) -> List[Optional[int]]:
    preds = []
    for row in ds.rows:
        preds.append(predict_row_core(rs, row))
    return preds

def count_undecided_and_errors(preds: List[Optional[int]], labels: List[int]) -> Tuple[int, int]:
    undec = 0
    errs = 0
    for p, y in zip(preds, labels):
        if p is None:
            undec += 1
        elif p != y:
            errs += 1
    return undec, errs

def mdl_lengths(rs: RuleSet, ds: Dataset, alpha: float, beta: float, gamma: float, eta: float, xi: float) -> Tuple[float, float, float, int, int]:
    """Returns: L_total, L_model, L_data, undecided, errors"""
    L_model = 0.0
    for r in rs.rules:
        b = rule_body_size(r)
        c = rule_cov_count(r)
        L_model += (alpha + beta * b + gamma * math.log(1 + c))

    preds = predict_all_core(rs, ds)
    undec, errs = count_undecided_and_errors(preds, ds.labels)
    L_data = eta * undec + xi * errs
    return (L_model + L_data, L_model, L_data, undec, errs)

def dedup_candidates(cands: List[Rule]) -> List[Rule]:
    """Keep one representative per (body, label)."""
    seen = set()
    out = []
    for r in cands:
        key = (tuple(sorted(r.body)), r.label)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def ec_mdl_select(pool: RuleSet, ds: Dataset, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1, eta: float = 2.0, xi: float = 4.0, verbose: bool = True) -> Tuple[RuleSet, List[Tuple[int, float, float]]]:
    """Greedy forward selection to minimize EC-MDL objective."""
    cands = dedup_candidates(pool.rules)
    selected = RuleSet([])
    L_sel, Lm, Ld, U, E = mdl_lengths(selected, ds, alpha, beta, gamma, eta, xi)
    if verbose:
        print(f"[mdl] start: K=0 L={L_sel:.3f} (model={Lm:.3f}, data={Ld:.3f}; undec={U}, err={E})")
    used = [False] * len(cands)
    log: List[Tuple[int, float, float]] = []
    while True:
        best_i = -1
        best_L = L_sel
        best_delta = 0.0
        for i, r in enumerate(cands):
            if used[i]:
                continue
            trial = RuleSet(selected.rules + [r])
            L_trial, *_ = mdl_lengths(trial, ds, alpha, beta, gamma, eta, xi)
            if L_trial < best_L - 1e-12:
                best_L = L_trial
                best_i = i
                best_delta = L_trial - L_sel
        if best_i == -1:
            if verbose:
                print(f"[mdl] stop: no rule reduces L further. K={len(selected.rules)} L={L_sel:.3f}")
            break
        used[best_i] = True
        selected = RuleSet(selected.rules + [cands[best_i]])
        L_sel = best_L
        log.append((best_i, best_delta, L_sel))
        if verbose:
            r = cands[best_i]
            b = rule_body_size(r)
            c = rule_cov_count(r)
            prec = rule_precision(r, ds)
            print(f"[mdl] +rule pool_idx={best_i:02d} body_size={b} cov={c} prec={prec:.3f} ΔL={best_delta:.3f} K={len(selected.rules)} L={L_sel:.3f}")
    return selected, log
