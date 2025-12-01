# cv.py — ECCRS CV with per-fold rule saving
import argparse, random, time, json, csv, math, os, statistics as stats
from typing import List, Tuple, Dict, Any
import numpy as np

# metrics
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif

from eccrs.data import load_lp, Dataset
from eccrs.trainer import (
    train_cf_eccrs,
    prune_redundant_same_label,
    audit_eccrs_global,
    ensure_at_least_one_pair,
    laminar_strict_closure,
    evaluate_with_fallback,
)
from eccrs.model import RuleSet
from eccrs import fallback as fb  # for nearest-rule completion


# -----------------------
# Helpers: dataset subset
# -----------------------
def subset(ds: Dataset, keep_idx: List[int]) -> Dataset:
    rows = [ds.rows[i] for i in keep_idx]
    labels = [ds.labels[i] for i in keep_idx]
    sub = Dataset(rows=rows, labels=labels, features=ds.features, label_attr=ds.label_attr)
    # keep pretty names if present
    if hasattr(ds, "feat_names"):
        try:
            sub.feat_names = ds.feat_names
        except Exception:
            pass
    return sub


# -----------------------
# Build dense binary matrix from Dataset
# -----------------------
def matrix_from_dataset(ds: Dataset) -> np.ndarray:
    """
    Returns X (n_samples, n_features) binary/int matrix.
    Works if:
    - each row is a dict-like mapping feature -> 0/1, or
    - a list/tuple aligned with ds.features, or
    - a set of "active" features (membership as 1).
    """
    feats = ds.features
    n, d = len(ds.rows), len(feats)
    X = np.zeros((n, d), dtype=float)
    for i, row in enumerate(ds.rows):
        if isinstance(row, dict):
            for j, f in enumerate(feats):
                X[i, j] = float(row.get(f, 0))
        elif isinstance(row, (list, tuple)) and len(row) == d:
            X[i, :] = np.asarray(row, dtype=float)
        elif isinstance(row, set):
            act = row
            for j, f in enumerate(feats):
                X[i, j] = 1.0 if f in act else 0.0
        else:
            try:
                for j, f in enumerate(feats):
                    X[i, j] = float(row.get(f, 0))
            except Exception:
                raise TypeError("Unsupported row structure for matrix conversion.")
    return X


# -----------------------
# Stratified split builders (pure Python, no sklearn)
# -----------------------
def stratified_kfold_indices(labels: List[int], k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    folds_pos = [[] for _ in range(k)]
    folds_neg = [[] for _ in range(k)]
    for t, i in enumerate(pos):
        folds_pos[t % k].append(i)
    for t, i in enumerate(neg):
        folds_neg[t % k].append(i)
    folds = []
    all_idx = set(range(len(labels)))
    for f in range(k):
        test_idx = sorted(folds_pos[f] + folds_neg[f])
        train_idx = sorted(list(all_idx - set(test_idx)))
        folds.append((train_idx, test_idx))
    return folds


def stratified_shuffle_splits(labels: List[int], test_size: float, repeats: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    assert 0.0 < test_size < 1.0, "test_size must be in (0,1)"
    pos_all = [i for i, y in enumerate(labels) if y == 1]
    neg_all = [i for i, y in enumerate(labels) if y == 0]
    folds = []
    for r in range(repeats):
        rnd = random.Random(seed + 1000 * r)
        pos = pos_all[:]; neg = neg_all[:]
        rnd.shuffle(pos); rnd.shuffle(neg)
        n_pos = len(pos); n_neg = len(neg)
        n_test_pos = max(1, min(n_pos - 1, int(round(n_pos * test_size))))
        n_test_neg = max(1, min(n_neg - 1, int(round(n_neg * test_size))))
        test_idx = sorted(pos[:n_test_pos] + neg[:n_test_neg])
        train_idx = sorted(pos[n_test_pos:] + neg[n_test_neg:])
        folds.append((train_idx, test_idx))
    return folds


# -----------------------
# Rule stats
# -----------------------
def avg_body_size(rs: RuleSet) -> float:
    if not rs.rules:
        return 0.0
    return float(sum(len(getattr(r, "body", [])) for r in rs.rules)) / float(len(rs.rules))


def unique_features_used(rs: RuleSet) -> int:
    """
    Best-effort: assumes each rule r.body is a list of literals, and each literal has an attribute id
    accessible as lit.attr or lit[0]. Falls back to 0 if unsupported.
    """
    feats = set()
    try:
        for r in rs.rules:
            for lit in getattr(r, "body", []):
                if hasattr(lit, "attr"):
                    feats.add(lit.attr)
                elif isinstance(lit, (list, tuple)) and len(lit) > 0:
                    feats.add(lit[0])
                else:
                    s = str(lit)
                    if s.startswith("a("):
                        feats.add(s.split(",")[0])
    except Exception:
        return 0
    return len(feats)


# -----------------------
# Calibration utilities
# -----------------------
def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return float(np.mean((probs - y_true) ** 2))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs >= lo) & (probs < hi) if b < n_bins - 1 else (probs >= lo) & (probs <= hi)
        nk = np.sum(mask)
        if nk == 0:
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (nk / N) * abs(acc - conf)
    return float(ece)


# -----------------------
# Fallback builder (kNN)
# -----------------------
def build_knn_fallback(X_train: np.ndarray, y_train: np.ndarray, k: int, weighted: bool, feat_weight: str):
    weights = 'distance' if weighted else 'uniform'
    metric = 'hamming'
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)

    # optional simple MI reweighting: scale columns by MI^(1/2)
    if feat_weight == "mi":
        try:
            mi = mutual_info_classif(X_train, y_train, discrete_features=True)
            scale = np.sqrt(np.clip(mi, 1e-12, None))
            X_train = X_train * scale
            scaler = scale
        except Exception:
            scaler = None
    else:
        scaler = None

    knn.fit(X_train, y_train)
    return knn, scaler


def knn_predict_proba(knn: KNeighborsClassifier, X: np.ndarray, scaler: np.ndarray | None):
    if scaler is not None:
        X = X * scaler
    return knn.predict_proba(X)[:, 1]


# -----------------------
# Rule saving utilities
# -----------------------
def _feat_name(ds: Dataset, j) -> str:
    """
    Try to recover a pretty feature name for id/index j.
    Supports ds.feat_names as dict or list; falls back to 'a(j)'.
    """
    try:
        if hasattr(ds, "feat_names"):
            fn = ds.feat_names
            if isinstance(fn, dict):
                return str(fn.get(j, f"a({j})"))
            if isinstance(fn, (list, tuple)):
                idx = j - 1 if (isinstance(j, int) and 1 <= j <= len(fn)) else j
                if isinstance(idx, int) and 0 <= idx < len(fn):
                    return str(fn[idx])
    except Exception:
        pass
    return f"a({j})"


def _normalize_body(body):
    """
    Return a sorted list of (feat_id, value) from many possible body shapes:
    dict{feat->val}, set/list of (feat,val), set/list of feat (assume val=1), etc.
    """
    pairs = []
    if body is None:
        return pairs

    if isinstance(body, dict):
        for j, v in body.items():
            try:
                pairs.append((int(j), int(v)))
            except Exception:
                pass
    else:
        try:
            it = list(body)
        except Exception:
            return pairs
        for lit in it:
            if isinstance(lit, (tuple, list)) and len(lit) >= 2:
                j, v = lit[0], lit[1]
                try:
                    pairs.append((int(j), int(v)))
                except Exception:
                    pass
            elif isinstance(lit, (int, np.integer)):
                pairs.append((int(lit), 1))
            else:
                s = str(lit)
                if s.startswith("a(") and "=" in s:
                    try:
                        left, right = s.split("=")
                        v = int(right.strip())
                        jtxt = left[left.find("(")+1:left.find(")")]
                        j = int(jtxt)
                        pairs.append((j, v))
                    except Exception:
                        pass

    pairs = sorted(list({(j, v) for j, v in pairs}), key=lambda t: (t[0], t[1]))
    return pairs


def save_ruleset_for_fold(rs: RuleSet, ds: Dataset, out_json_path: str, out_txt_path: str):
    """
    Writes:
    - JSON: [{"label": int, "body": [{"feat":id,"name":..., "val":0/1}], "meta":{...}}, ...]
    - TXT: 'IF name1=1 & name2=0 THEN y=1'
    """
    rules = None
    for attr in ("rules", "rules_", "rule_list", "rule_list_"):
        if hasattr(rs, attr):
            rules = getattr(rs, attr)
            break
    if rules is None:
        try:
            rules = list(rs)
        except Exception:
            rules = []

    json_rules = []
    lines = []
    for r in rules:
        body = getattr(r, "body", None)
        if body is None and hasattr(r, "__dict__"):
            for nm in ("B", "antecedents", "conds", "conditions", "terms"):
                if nm in r.__dict__:
                    body = r.__dict__[nm]
                    break
        y = getattr(r, "label", None)
        if y is None and hasattr(r, "__dict__"):
            for nm in ("y", "cls", "consequent", "label_"):
                if nm in r.__dict__:
                    y = r.__dict__[nm]
                    break
        if y is True:
            y = 1
        if y is False:
            y = 0
        try:
            y = int(y)
        except Exception:
            y = None

        pairs = _normalize_body(body)
        body_items = [{"feat": j, "name": _feat_name(ds, j), "val": v} for (j, v) in pairs]

        meta = {}
        for key in ("precision", "recall", "support", "tp", "fp", "tn", "fn", "cov"):
            if hasattr(r, key):
                try:
                    meta[key] = float(getattr(r, key))
                except Exception:
                    pass

        json_rules.append({"label": y, "body": body_items, "meta": meta})
        conds = " & ".join([f"{bi['name']}={bi['val']}" for bi in body_items]) if body_items else "(TRUE)"
        lines.append(f"IF {conds} THEN y={y}")

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_rules, f, indent=2, ensure_ascii=False)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------
# Per-fold run
# -----------------------
def run_one_fold(ds_full: Dataset, train_idx: List[int], test_idx: List[int], args, fold_id: int | None = None) -> Dict[str, Any]:
    ds_train = subset(ds_full, train_idx)
    ds_test = subset(ds_full, test_idx)

    # Dense matrices for fallback
    X_train = matrix_from_dataset(ds_train)
    X_test = matrix_from_dataset(ds_test)
    y_train = np.asarray(ds_train.labels, dtype=int)
    y_test = np.asarray(ds_test.labels, dtype=int)

    # Train ECCRS
    t0 = time.perf_counter()
    rs = train_cf_eccrs(
        ds_train,
        max_iters=args.max_iters,
        verbose=False,
        max_rules=args.max_rules,
        min_cov_pos=args.min_cov_pos,
        min_gain=args.min_gain,
    )
    rs = prune_redundant_same_label(ds_train, rs)
    ok, msg = audit_eccrs_global(ds_train, rs)
    print(f"[audit] train ECCRS: {'OK' if ok else 'VIOLATION'}{'' if ok else ' - ' + msg}")

    # Optional selector
    if args.selector == "mdl":
        from eccrs.trainer import run_mdl_selector
        rs_sel = run_mdl_selector(rs, ds_train, c0=args.mdl_c0, c1=args.mdl_c1, eta=args.mdl_eta, verbose=False)
    elif args.selector == "fallback_gain":
        from eccrs.trainer import run_fallback_gain_selector
        rs_sel = run_fallback_gain_selector(
            rs, ds_train,
            fallback=args.fallback, k=args.k, train_ref=ds_train,
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
            max_add=args.fg_max_rules, min_delta=args.fg_min_delta, verbose=False
        )
    else:
        rs_sel = None

    if rs_sel is not None and args.use_selected:
        rs = rs_sel

    rs = ensure_at_least_one_pair(rs, ds_train, verbose=False)

    # Laminar strict closure
    t1 = time.perf_counter()
    if args.laminar_strict:
        rs = laminar_strict_closure(
            rs, ds_train,
            verbose=False,
            laminar_min_overlap=args.laminar_min_overlap,
            orient_by_overlap=(not args.laminar_no_overlap_majority),
        )
    t_strict1 = time.perf_counter()

    # Save rules for this fold (after closure/selection)
    if args.save_rules_dir and fold_id is not None:
        json_path = os.path.join(args.save_rules_dir, f"fold_{fold_id:02d}.json")
        txt_path = os.path.join(args.save_rules_dir, f"fold_{fold_id:02d}.txt")
        try:
            save_ruleset_for_fold(rs, ds_train, json_path, txt_path)
            print(f"[rules] saved {json_path} and {txt_path}")
        except Exception as e:
            print(f"[rules] failed to save rules for fold {fold_id}: {e}")

    # Rule-only predictions (with abstentions)
    rule_preds = rs.predict_all(ds_test)  # list of 0/1 or None
    covered_mask = np.array([p is not None for p in rule_preds], dtype=bool)
    y_rule_pred = np.array([int(p) if p is not None else -1 for p in rule_preds], dtype=int)

    # Fill preds/probs depending on fallback
    y_prob = np.zeros_like(y_test, dtype=float)
    y_pred = np.zeros_like(y_test, dtype=int)

    if args.fallback == "knn":
        knn, scaler = build_knn_fallback(
            X_train, y_train, k=args.k,
            weighted=args.knn_weighted,
            feat_weight=args.feat_weight
        )
        knn_probs = knn_predict_proba(knn, X_test, scaler)

        for i, covered in enumerate(covered_mask):
            if covered:
                y_pred[i] = y_rule_pred[i]
                y_prob[i] = 1.0 if y_rule_pred[i] == 1 else 0.0
            else:
                y_prob[i] = knn_probs[i]
                y_pred[i] = 1 if y_prob[i] >= 0.5 else 0

    elif args.fallback == "nearest":
        # symbolic nearest rule completion for abstentions
        for i, covered in enumerate(covered_mask):
            if covered:
                y_pred[i] = y_rule_pred[i]
                y_prob[i] = 1.0 if y_rule_pred[i] == 1 else 0.0
            else:
                row = ds_test.rows[i]
                row_lits = {(j, v) for j, v in row.items()}
                lab = fb.nearest_rule_completion(rs, ds_train, row_lits, label_pref=1)
                y_pred[i] = lab
                # crisp probability proxy
                y_prob[i] = 1.0 if lab == 1 else 0.0

    else:
        # abstentions remain; neutral 0.5 for abstained points
        for i, covered in enumerate(covered_mask):
            if covered:
                y_pred[i] = y_rule_pred[i]
                y_prob[i] = 1.0 if y_rule_pred[i] == 1 else 0.0
            else:
                y_prob[i] = 0.5
                y_pred[i] = 0

    n_fb = int(np.sum(~covered_mask))
    print(f"[cv] fallback used on {n_fb} of {len(y_test)} test rows with mode={args.fallback}")

    # final metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # Handle AUCs safely when only one class present
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception:
        pr_auc = float("nan")

    br = brier_score(y_prob, y_test)
    ece10 = expected_calibration_error(y_prob, y_test, n_bins=10)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # selective metrics
    n = len(y_test)
    cov = float(np.mean(covered_mask)) if n else 0.0
    sel_acc = float(np.mean(y_test[covered_mask] == y_pred[covered_mask])) if np.any(covered_mask) else float("nan")
    fb_cov = 1.0 - cov
    fb_acc = float(np.mean(y_test[~covered_mask] == y_pred[~covered_mask])) if np.any(~covered_mask) else float("nan")

    # structural / complexity
    n_rules = len(rs.rules)
    avg_body = avg_body_size(rs)
    try:
        n_feats = unique_features_used(rs)
    except Exception:
        n_feats = 0

    # placeholders for ECCRS-native exception semantics (needs API hooks)
    max_exc_depth = float("nan")
    override_rate = float("nan")
    conflict_abstentions = float("nan")
    overlap_index = float("nan")

    t2 = time.perf_counter()

    # Sanity check against evaluate_with_fallback
    try:
        em = evaluate_with_fallback(
            rs, ds_test,
            fallback=args.fallback, k=args.k, train_ref=ds_train,
            tag="test_after_strict",
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
        )
        if "overall_acc" in em and abs(em["overall_acc"] - acc) > 1e-6:
            print(f"[warn] our overall acc {acc:.4f} differs from evaluate_with_fallback {em['overall_acc']:.4f}")
        if "selective_acc" in em and np.isfinite(sel_acc):
            if abs(em["selective_acc"] - sel_acc) > 1e-6:
                print(f"[warn] our selective acc {sel_acc:.4f} differs from evaluate_with_fallback {em['selective_acc']:.4f}")
    except Exception:
        pass

    print(
        f"[cv] Fold train={len(train_idx):5d} test={len(test_idx):5d} "
        f"acc={acc:.3f} bal={bal_acc:.3f} f1M={f1_macro:.3f} auc={roc_auc if not math.isnan(roc_auc) else float('nan'):.3f} "
        f"cov={cov:.3f} sel={sel_acc if not math.isnan(sel_acc) else float('nan'):.3f} "
        f"rules={n_rules} body={avg_body:.2f} feats={n_feats} "
        f"time(train/strict/total)={(t1-t0):.2f}/{(t_strict1-t1):.2f}/{(t2-t0):.2f}s"
    )

    return dict(
        # predictive
        acc=acc, bal_acc=bal_acc, f1_macro=f1_macro, roc_auc=roc_auc, pr_auc=pr_auc, brier=br, ece10=ece10,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        # selective
        coverage_rule=cov, selective_acc=sel_acc, fallback_coverage=fb_cov, fallback_acc=fb_acc,
        # complexity
        n_rules=n_rules, avg_body=avg_body, n_features=n_feats,
        max_exception_depth=max_exc_depth, override_rate=override_rate,
        conflict_abstentions=conflict_abstentions, overlap_index=overlap_index,
        # timings
        time_train=(t1 - t0), time_strict=(t_strict1 - t1), time_total=(t2 - t0),
    )


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Cross-Validation runner for ECCRS (k-fold and stratified shuffle).")

    # Data
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--label_attr", type=int, default=10)
    ap.add_argument("--ignore_attr", type=int, action="append", default=[])

    # CV controls
    ap.add_argument("--cv_mode", type=str, default="kfold", choices=["kfold", "shuffle"])
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_folds", type=str, default=None)
    ap.add_argument("--load_folds", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default="cv_results.csv")

    # Training
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--max_rules", type=int, default=10**9)
    ap.add_argument("--min_cov_pos", type=int, default=1)
    ap.add_argument("--min_gain", type=int, default=1)

    # Selector
    ap.add_argument("--selector", type=str, default="none", choices=["none", "mdl", "fallback_gain"])
    ap.add_argument("--mdl_c0", type=float, default=2.0)
    ap.add_argument("--mdl_c1", type=float, default=1.0)
    ap.add_argument("--mdl_eta", type=float, default=2.0)
    ap.add_argument("--fg_max_rules", type=int, default=100)
    ap.add_argument("--fg_min_delta", type=float, default=0.0)
    ap.add_argument("--use_selected", action="store_true")

    # Fallbacks
    ap.add_argument("--fallback", type=str, default="knn", choices=["abst", "nearest", "knn"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--knn_weighted", action="store_true")
    ap.add_argument("--feat_weight", type=str, default="none", choices=["none", "mi"])

    # Laminar strict closure
    ap.add_argument("--laminar_strict", action="store_true")
    ap.add_argument("--laminar_min_overlap", type=int, default=1)
    ap.add_argument("--laminar_no_overlap_majority", action="store_true")

    # save per-fold rules
    ap.add_argument("--save_rules_dir", type=str, default=None, help="If set, dump the learned ECCRS rule set for each fold (JSON + TXT).")

    args = ap.parse_args()

    # Load data
    ignore_set = set(args.ignore_attr) if args.ignore_attr else set()
    print(f"[load] file={args.data}, label_attr=a({args.label_attr}), ignore={sorted(ignore_set)}")
    ds_full: Dataset = load_lp(args.data, label_attr=args.label_attr, ignore_attrs=ignore_set)
    print(f"[data] rows={ds_full.n}, features={len(ds_full.features)}")

    # Build or load splits
    folds: List[Tuple[List[int], List[int]]] = []
    if args.load_folds:
        with open(args.load_folds, "r", encoding="utf-8") as f:
            obj = json.load(f)
            folds = [(sorted(x["train_idx"]), sorted(x["test_idx"])) for x in obj["folds"]]
        print(f"[cv] Loaded {len(folds)} predefined splits from {args.load_folds}")
    else:
        if args.cv_mode == "kfold":
            for r in range(args.repeats):
                folds.extend(stratified_kfold_indices(ds_full.labels, args.kfolds, seed=args.seed + 1000 * r))
        else:
            folds = stratified_shuffle_splits(ds_full.labels, test_size=args.test_size, repeats=args.repeats, seed=args.seed)

        if args.save_folds:
            to_save = {
                "meta": {
                    "mode": args.cv_mode,
                    "kfolds": args.kfolds,
                    "repeats": args.repeats,
                    "test_size": args.test_size,
                    "seed": args.seed,
                    "label_attr": args.label_attr,
                    "ignore": sorted(ignore_set),
                },
                "folds": [{"train_idx": tr, "test_idx": te} for (tr, te) in folds]
            }
            with open(args.save_folds, "w", encoding="utf-8") as f:
                json.dump(to_save, f)
            print(f"[cv] Saved {len(folds)} splits to {args.save_folds}")

    # Run all folds
    rows: List[Dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n[cv] Fold {fold_id}/{len(folds)} train={len(train_idx)} test={len(test_idx)}")
        metrics = run_one_fold(ds_full, train_idx, test_idx, args, fold_id=fold_id)
        rows.append({"fold": fold_id, **metrics})

    # Aggregate
    def mean_std(key: str):
        vals = [r[key] for r in rows if (r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key])))]
        if not vals:
            return float("nan"), float("nan")
        mu = stats.mean(vals)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        return mu, sd

    keys_to_print = [
        ("acc", "Accuracy"),
        ("bal_acc", "Balanced acc"),
        ("f1_macro", "F1-macro"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier", "Brier"),
        ("ece10", "ECE@10"),
        ("coverage_rule", "Coverage (rule)"),
        ("selective_acc", "Selective acc"),
        ("fallback_coverage", "Coverage (fallback)"),
        ("fallback_acc", "Fallback acc"),
        ("n_rules", "#Rules"),
        ("avg_body", "Avg body len"),
        ("n_features", "#Features"),
        ("time_train", "Train time (s)"),
        ("time_strict", "Laminar time (s)"),
        ("time_total", "Total time (s)"),
    ]
    print("\n================ CV SUMMARY ================")
    for key, label in keys_to_print:
        mu, sd = mean_std(key)
        print(f" {label:22s}: {mu:.3f} ± {sd:.3f}")
    print("===========================================")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            # predictive
            "acc", "bal_acc", "f1_macro", "roc_auc", "pr_auc", "brier", "ece10", "tn", "fp", "fn", "tp",
            # selective
            "coverage_rule", "selective_acc", "fallback_coverage", "fallback_acc",
            # complexity
            "n_rules", "avg_body", "n_features", "max_exception_depth", "override_rate", "conflict_abstentions", "overlap_index",
            # timings
            "time_train", "time_strict", "time_total",
        ])
        for r in rows:
            writer.writerow([
                r["fold"],
                f"{r['acc']:.6f}",
                f"{r['bal_acc']:.6f}",
                f"{r['f1_macro']:.6f}",
                f"{r['roc_auc']:.6f}" if not math.isnan(r['roc_auc']) else "nan",
                f"{r['pr_auc']:.6f}" if not math.isnan(r['pr_auc']) else "nan",
                f"{r['brier']:.6f}",
                f"{r['ece10']:.6f}",
                r["tn"], r["fp"], r["fn"], r["tp"],
                f"{r['coverage_rule']:.6f}",
                f"{r['selective_acc']:.6f}" if not math.isnan(r['selective_acc']) else "nan",
                f"{r['fallback_coverage']:.6f}",
                f"{r['fallback_acc']:.6f}" if not math.isnan(r['fallback_acc']) else "nan",
                f"{r['n_rules']}",
                f"{r['avg_body']:.6f}",
                f"{r['n_features']}",
                "nan", "nan", "nan", "nan",
                f"{r['time_train']:.6f}",
                f"{r['time_strict']:.6f}",
                f"{r['time_total']:.6f}",
            ])
    print(f"\n[cv] wrote per-fold metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
