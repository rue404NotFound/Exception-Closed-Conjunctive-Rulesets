# eccrs/fallback.py
# Π-completion (nearest-rule completion) and k-NN fallback.
from typing import Optional, Tuple, List
from .data import Dataset
from .model import RuleSet, Rule

def popcount(x: int) -> int:
    return x.bit_count()

# ---------- Π-completion (nearest rule by literals to add) ----------

def rule_precision_on_train(r: Rule, ds_tr: Dataset) -> float:
    cov = r.cov_bits
    if cov == 0:
        return 0.5
    pos = popcount(cov & ds_tr.Y_pos)
    neg = popcount(cov & ds_tr.Y_neg)
    tot = pos + neg
    if tot == 0:
        return 0.5
    if r.label == 1:
        return pos / tot
    else:
        return neg / tot

def nearest_rule_completion(rs: RuleSet,
                            ds_train: Dataset,
                            row_literals: set,
                            label_pref: int = 1) -> int:
    """
    Among rules, choose ones with minimal |body \ L(x)|.
    Tie-breakers: (i) higher training precision, (ii) larger body,
                  (iii) larger training coverage, (iv) label_pref.
    """
    best_rules: List[Tuple[int, Rule]] = []  # (miss_count, rule)
    min_miss = None

    for r in rs.rules:
        miss = len([lit for lit in r.body if lit not in row_literals])
        if min_miss is None or miss < min_miss:
            min_miss = miss
            best_rules = [(miss, r)]
        elif miss == min_miss:
            best_rules.append((miss, r))

    # tie-break
    def key_fn(item):
        _, r = item
        prec = rule_precision_on_train(r, ds_train)
        cov = popcount(r.cov_bits)
        return (-prec, -len(r.body), -cov, 0 if r.label == label_pref else 1)

    chosen = sorted(best_rules, key=key_fn)[0][1]
    return chosen.label

# ---------- k-NN over training rows (Hamming over binary features) ----------

def hamming_row_distance(x_row: dict, y_row: dict, features: List[int]) -> int:
    d = 0
    for j in features:
        x = x_row.get(j, 0)
        y = y_row.get(j, 0)
        d += (1 if x != y else 0)
    return d

def knn_predict(ds_train: Dataset,
                row: dict,
                k: int = 5,
                label_pref: int = 1,
                exclude_index: Optional[int] = None) -> int:
    """
    Simple unweighted k-NN on binary features (Hamming).
    Uses ONLY training set (optionally leave-one-out by excluding index).
    Deterministic tie break with label_pref.
    """
    n = ds_train.n
    if n == 0:
        # Shouldn't happen, but default to label_pref
        return label_pref

    # Gather (distance, idx, label)
    dist_list: List[Tuple[int, int, int]] = []
    for i, (r, y) in enumerate(zip(ds_train.rows, ds_train.labels)):
        if exclude_index is not None and i == exclude_index:
            continue
        d = hamming_row_distance(row, r, ds_train.features)
        dist_list.append((d, i, y))

    if not dist_list:
        # e.g., LOO with single-row train set
        return label_pref

    dist_list.sort(key=lambda t: (t[0], t[1]))  # stable, deterministic
    k_eff = min(k, len(dist_list))
    top = dist_list[:k_eff]

    votes_pos = sum(1 for _, _, y in top if y == 1)
    votes_neg = k_eff - votes_pos
    if votes_pos > votes_neg:
        return 1
    if votes_neg > votes_pos:
        return 0
    # tie -> label_pref
    return label_pref

# ---------- Unified fallback entry ----------

def predict_with_fallback(rs: RuleSet,
                          ds_train: Dataset,
                          row: dict,
                          mode: str,
                          k: int = 5,
                          label_pref: int = 1,
                          exclude_index: Optional[int] = None) -> int:
    """
    mode in {'nearest','knn'}; used ONLY when core ECCRS abstains.
    """
    lits = {(j, v) for j, v in row.items()}
    if mode == "nearest":
        return nearest_rule_completion(rs, ds_train, lits, label_pref=label_pref)
    elif mode == "knn":
        return knn_predict(ds_train, row, k=k, label_pref=label_pref, exclude_index=exclude_index)
    else:
        # Shouldn't happen; default safe choice
        return label_pref
