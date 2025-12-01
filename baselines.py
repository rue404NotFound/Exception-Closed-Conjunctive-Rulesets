import argparse, json, math, random, time, csv, re, warnings
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB

# Warnings: quiet down common library noise
warnings.filterwarnings(
    "ignore",
    message=r"This use of ``\*`` has resulted in matrix multiplication\.",
    category=UserWarning,
    module=r"cvxpy"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"cloudpickle")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"joblib")

# ----- ECCRS dataset loader -----
from eccrs.data import load_lp, Dataset


# -----------------------
# Helpers: dataset to matrix
# -----------------------

def subset(ds: Dataset, keep_idx: List[int]) -> Dataset:
    rows = [ds.rows[i] for i in keep_idx]
    labels = [ds.labels[i] for i in keep_idx]
    sub = Dataset(rows=rows, labels=labels, features=ds.features, label_attr=ds.label_attr)
    if hasattr(ds, "feat_names"):
        try:
            sub.feat_names = ds.feat_names
        except Exception:
            pass
    return sub

def matrix_from_dataset(ds: Dataset) -> np.ndarray:
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
            for j, f in enumerate(feats):
                X[i, j] = 1.0 if f in row else 0.0
        else:
            try:
                for j, f in enumerate(feats):
                    X[i, j] = float(row.get(f, 0))
            except Exception:
                raise TypeError("Unsupported row structure for matrix conversion.")
    return X

def stratified_kfold_indices(labels: List[int], k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rnd = random.Random(seed)
    rnd.shuffle(pos); rnd.shuffle(neg)
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
    assert 0.0 < test_size < 1.0
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
# Calibration metrics
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
# BRCG safe column prep
# -----------------------

def _sanitize_cols_for_brcg(cols: List[str]) -> List[str]:
    safe = []
    for c in cols:
        c = re.sub(r'[^A-Za-z0-9_]', '_', str(c))
        safe.append(f"{c}=1")
    return safe

def prepare_brcg_dataframe(X_numeric: np.ndarray, colnames: List[str]) -> pd.DataFrame:
    """Boolean DataFrame with explicit =1 columns for AIX360 BRCG."""
    safe_cols = _sanitize_cols_for_brcg(colnames)
    return pd.DataFrame(X_numeric.astype(bool), columns=safe_cols)


# -----------------------
# Model registry
# -----------------------

@dataclass
class ModelSpec:
    key: str
    ctor: Callable[[], Any]             # returns an unfitted estimator
    supports_proba: bool
    family: str                         # 'tree' | 'list' | 'set' | 'linear' | 'knn' | 'other'

def make_registry(seed: int = 42) -> Dict[str, ModelSpec]:
    reg: Dict[str, ModelSpec] = {}

    # Trees
    reg["dt_gini"] = ModelSpec("dt_gini",
        ctor=lambda: DecisionTreeClassifier(criterion="gini", random_state=seed),
        supports_proba=True, family="tree")
    reg["dt_entropy"] = ModelSpec("dt_entropy",
        ctor=lambda: DecisionTreeClassifier(criterion="entropy", random_state=seed),
        supports_proba=True, family="tree")

    # RIPPER
    def ripper_ctor():
        try:
            from wittgenstein import RIPPER
        except Exception as e:
            raise ImportError(f"wittgenstein not installed: {e}")
        return RIPPER(random_state=seed)
    reg["ripper"] = ModelSpec("ripper", ctor=ripper_ctor, supports_proba=True, family="list")

    # AIX360 BRL
    def brl_aix_ctor():
        try:
            from aix360.algorithms.rbm import BayesianRuleListClassifier
        except Exception as e:
            raise ImportError(f"aix360 not installed or BRL missing: {e}")
        return BayesianRuleListClassifier(random_state=seed)
    reg["brl_aix360"] = ModelSpec("brl_aix360", ctor=brl_aix_ctor, supports_proba=True, family="list")

    # AIX360 BRCG
    def brcg_ctor():
        try:
            from aix360.algorithms.rbm import BooleanRuleCG
        except Exception as e:
            raise ImportError(f"aix360 not installed or BRCG missing: {e}")
        return BooleanRuleCG()
    reg["brcg"] = ModelSpec("brcg", ctor=brcg_ctor, supports_proba=True, family="set")

    # imodels BRL
    def brl_imodels_ctor():
        try:
            from imodels import BayesianRuleListClassifier
        except Exception as e:
            raise ImportError(f"imodels BRL not available: {e}")
        return BayesianRuleListClassifier()
    reg["brl_imodels"] = ModelSpec("brl_imodels", ctor=brl_imodels_ctor, supports_proba=True, family="list")

    # imodels Greedy Rule List
    def grl_ctor():
        try:
            from imodels import GreedyRuleListClassifier
        except Exception as e:
            raise ImportError(f"imodels GreedyRuleList not available: {e}")
        try:
            return GreedyRuleListClassifier(random_state=seed)
        except TypeError:
            return GreedyRuleListClassifier()
    reg["grl"] = ModelSpec("grl", ctor=grl_ctor, supports_proba=True, family="list")

    # imodels RuleFit
    def rulefit_ctor():
        try:
            from imodels import RuleFitClassifier
        except Exception as e:
            raise ImportError(f"imodels RuleFit not available: {e}")
        return RuleFitClassifier(max_rules=200, random_state=seed)
    reg["rulefit"] = ModelSpec("rulefit", ctor=rulefit_ctor, supports_proba=True, family="other")

    # Optional CORELS
    def corels_ctor():
        try:
            from corels import CorelsClassifier
        except Exception as e:
            raise ImportError(f"pycorels not installed: {e}")
        return CorelsClassifier()
    reg["corels"] = ModelSpec("corels", ctor=corels_ctor, supports_proba=False, family="list")

    # Linear and friends
    reg["logreg"] = ModelSpec("logreg",
        ctor=lambda: LogisticRegression(max_iter=1000, solver="lbfgs"),
        supports_proba=True, family="linear")

    reg["knn"] = ModelSpec("knn",
        ctor=lambda: KNeighborsClassifier(n_neighbors=15, metric="hamming", weights="distance"),
        supports_proba=True, family="knn")

    reg["rf"] = ModelSpec("rf",
        ctor=lambda: RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1),
        supports_proba=True, family="tree")

    reg["gbdt"] = ModelSpec("gbdt",
        ctor=lambda: GradientBoostingClassifier(random_state=seed),
        supports_proba=True, family="other")

    reg["hgbdt"] = ModelSpec("hgbdt",
        ctor=lambda: HistGradientBoostingClassifier(random_state=seed),
        supports_proba=True, family="other")

    reg["svm_linear"] = ModelSpec("svm_linear",
        ctor=lambda: SVC(kernel="linear", probability=True, random_state=seed),
        supports_proba=True, family="other")

    reg["svm_rbf"] = ModelSpec("svm_rbf",
        ctor=lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
        supports_proba=True, family="other")

    reg["nb_bern"] = ModelSpec("nb_bern",
        ctor=lambda: BernoulliNB(),
        supports_proba=True, family="other")

    reg["nb_gauss"] = ModelSpec("nb_gauss",
        ctor=lambda: GaussianNB(),
        supports_proba=True, family="other")

    # EBM
    def ebm_ctor():
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except Exception as e:
            raise ImportError(f"interpret not installed (EBM): {e}")
        return ExplainableBoostingClassifier(random_state=seed)
    reg["ebm"] = ModelSpec("ebm", ctor=ebm_ctor, supports_proba=True, family="other")

    return reg


# -----------------------
# Complexity extractors
# -----------------------

def complexity_tree(model) -> Dict[str, float]:
    try:
        node_count = getattr(model.tree_, "node_count", np.nan)
        depth = getattr(model.tree_, "max_depth", np.nan)
    except Exception:
        node_count = np.nan
        depth = np.nan
    return {"n_rules": np.nan, "avg_body": np.nan, "n_features": np.nan,
            "tree_nodes": float(node_count), "tree_depth": float(depth)}

def complexity_ripper(model) -> Dict[str, float]:
    n_rules = np.nan; avg_body = np.nan; n_feats = np.nan
    try:
        rs = getattr(model, "ruleset_", None)
        rules_list = getattr(rs, "rules", None) if rs is not None else None
        if rules_list is None:
            rules_list = getattr(model, "rules_", None)
        if rules_list is not None:
            lengths, feats = [], set()
            for r in rules_list:
                body = None
                for name in ("conds", "conditions", "antecedents", "body", "terms"):
                    if hasattr(r, name):
                        body = getattr(r, name); break
                if body is not None:
                    try:
                        lengths.append(len(body))
                        for lit in body:
                            if isinstance(lit, (tuple, list)) and len(lit)>0:
                                feats.add(lit[0])
                            elif hasattr(lit, "attr"):
                                feats.add(lit.attr)
                    except Exception:
                        pass
            if lengths:
                avg_body = float(np.mean(lengths))
                n_rules = float(len(lengths))
                n_feats = float(len(feats)) if feats else np.nan
    except Exception:
        pass
    return {"n_rules": n_rules, "avg_body": avg_body, "n_features": n_feats,
            "tree_nodes": np.nan, "tree_depth": np.nan}

def complexity_list_generic(model) -> Dict[str, float]:
    n_rules = np.nan; avg_body = np.nan; n_feats = np.nan
    try:
        rules = getattr(model, "rules_", None)
        if rules is None:
            rules = getattr(model, "rule_list_", None)
        if rules is None and hasattr(model, "get_rules"):
            try:
                rules = model.get_rules()
            except Exception:
                rules = None
        if rules is not None:
            lengths, feats = [], set()
            for r in rules:
                if isinstance(r, str):
                    body_str = r.split("THEN")[0]
                    k = body_str.count("&") + 1 if "IF" in body_str else np.nan
                    lengths.append(k)
                    tokens = [t.strip() for t in body_str.replace("IF", "").split("&")]
                    for t in tokens:
                        if "(" in t:
                            feats.add(t.split("(")[0].strip())
                        else:
                            toks = t.split()
                            if toks:
                                feats.add(toks[0])
                elif isinstance(r, (list, tuple)):
                    lengths.append(len(r))
                    for lit in r:
                        if isinstance(lit, (tuple, list)) and len(lit)>0:
                            feats.add(lit[0])
                else:
                    body = getattr(r, "body", None)
                    if body is not None:
                        lengths.append(len(body))
                        for lit in body:
                            if hasattr(lit, "attr"):
                                feats.add(lit.attr)
            if lengths:
                n_rules = float(len(lengths))
                avg_body = float(np.mean(lengths))
                n_feats = float(len(feats)) if feats else np.nan
    except Exception:
        pass
    return {"n_rules": n_rules, "avg_body": avg_body, "n_features": np.nan,
            "tree_nodes": np.nan, "tree_depth": np.nan}

def complexity_rulefit(model) -> Dict[str, float]:
    try:
        if hasattr(model, "get_rules"):
            df = model.get_rules()
            df = df[df["coef"].abs() > 1e-12] if "coef" in df.columns else df
            n_rules = float(len(df))
            if "rule" in df.columns:
                lengths = []
                for s in df["rule"]:
                    if isinstance(s, str):
                        lengths.append(s.count("&") + 1)
                avg_body = float(np.mean(lengths)) if lengths else np.nan
            else:
                avg_body = np.nan
            return {"n_rules": n_rules, "avg_body": avg_body, "n_features": np.nan,
                    "tree_nodes": np.nan, "tree_depth": np.nan}
    except Exception:
        pass
    return {"n_rules": np.nan, "avg_body": np.nan, "n_features": np.nan,
            "tree_nodes": np.nan, "tree_depth": np.nan}


# -----------------------
# Proba helper
# -----------------------

def safe_predict_proba(model, X_test):
    try:
        proba = model.predict_proba(X_test)
        if proba is None:
            return None
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                col = classes.index(1)
                return proba[:, col]
        return None
    except Exception:
        return None


# -----------------------
# Fit and eval per model and fold
# -----------------------

def eval_one_model_fold(model_key: str, spec: ModelSpec, X_train, y_train, X_test, y_test,
                        df_train=None, df_test=None, colnames: List[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        model = spec.ctor()
    except Exception as e:
        return {"_status": f"skipped: {e}"}

    # pandas DataFrame for certain models
    Xtr = df_train if model_key in ("brl_aix360", "brcg") and df_train is not None else X_train
    Xte = df_test  if model_key in ("brl_aix360", "brcg") and df_test  is not None else X_test

    try:
        if model_key in ("grl", "brl_imodels") and colnames is not None:
            try:
                model.fit(Xtr, y_train, feature_names=colnames)
            except TypeError:
                model.fit(Xtr, y_train)
        else:
            model.fit(Xtr, y_train)
    except Exception as e:
        return {"_status": f"fit_error: {e}"}
    t1 = time.perf_counter()

    try:
        y_pred = model.predict(Xte)
        y_pred = np.asarray(y_pred).astype(int)
    except Exception as e:
        return {"_status": f"predict_error: {e}"}
    t2 = time.perf_counter()

    y_prob = safe_predict_proba(model, Xte)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    if y_prob is not None:
        try:
            roc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc = float("nan")
        try:
            pr = average_precision_score(y_test, y_prob)
        except Exception:
            pr = float("nan")
        br = brier_score(y_prob, y_test)
        ece = expected_calibration_error(y_prob, y_test, n_bins=10)
    else:
        roc = float("nan"); pr = float("nan"); br = float("nan"); ece = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # Complexity
    if spec.family == "tree":
        comp = complexity_tree(model)
    elif model_key == "ripper":
        comp = complexity_ripper(model)
    elif model_key == "rulefit":
        comp = complexity_rulefit(model)
    elif spec.family in ("list", "set"):
        comp = complexity_list_generic(model)
    else:
        comp = {"n_rules": np.nan, "avg_body": np.nan, "n_features": np.nan,
                "tree_nodes": np.nan, "tree_depth": np.nan}

    return {
        "_status": "ok",
        "model": model_key,
        "acc": acc, "bal_acc": bal_acc, "f1_macro": f1m,
        "roc_auc": roc, "pr_auc": pr, "brier": br, "ece10": ece,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_rules": comp["n_rules"], "avg_body": comp["avg_body"],
        "n_features": comp["n_features"],
        "tree_nodes": comp["tree_nodes"], "tree_depth": comp["tree_depth"],
        "time_train": (t1 - t0), "time_pred": (t2 - t1), "time_total": (t2 - t0),
    }


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Baselines CV on .lp datasets")
    # Data
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--label_attr", type=int, required=True)
    ap.add_argument("--ignore_attr", type=int, action="append", default=[])
    # CV
    ap.add_argument("--load_folds", type=str, default=None)
    ap.add_argument("--cv_mode", type=str, choices=["kfold", "shuffle"], default="kfold")
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=8)
    # Models
    ap.add_argument(
        "--models",
        type=str,
        default="dt_gini,dt_entropy,ripper,brcg,brl_imodels,grl,rulefit,ebm,rf,gbdt,hgbdt,svm_linear,svm_rbf,nb_bern,nb_gauss,logreg,knn",
        help=("Comma separated keys from registry: "
              "dt_gini,dt_entropy,ripper,brl_aix360,brcg,brl_imodels,grl,rulefit,corels,"
              "logreg,knn,rf,gbdt,hgbdt,svm_linear,svm_rbf,nb_bern,nb_gauss,ebm")
    )
    # Output
    ap.add_argument("--out_csv", type=str, default="baselines_results.csv")

    args = ap.parse_args()

    ignore_set = set(args.ignore_attr) if args.ignore_attr else set()
    print(f"[load] file={args.data} label_attr=a({args.label_attr}) ignore={sorted(ignore_set)}")
    ds_full: Dataset = load_lp(args.data, label_attr=args.label_attr, ignore_attrs=ignore_set)
    print(f"[data] rows={ds_full.n} features={len(ds_full.features)}")

    X_all = matrix_from_dataset(ds_full)
    y_all = np.asarray(ds_full.labels, dtype=int)
    colnames = [str(f) for f in ds_full.features]

    # Two DataFrame views: default numeric and BRCG safe boolean with =1 columns
    df_all = pd.DataFrame(X_all, columns=colnames)
    df_all_brcg = prepare_brcg_dataframe(X_all, colnames)

    # Folds
    if args.load_folds:
        with open(args.load_folds, "r", encoding="utf-8") as f:
            obj = json.load(f)
        folds = [(sorted(x["train_idx"]), sorted(x["test_idx"])) for x in obj["folds"]]
        print(f"[cv] Loaded {len(folds)} predefined splits from {args.load_folds}")
    else:
        if args.cv_mode == "kfold":
            folds = []
            for r in range(args.repeats):
                folds.extend(stratified_kfold_indices(list(y_all), args.kfolds, seed=args.seed + 1000 * r))
        else:
            folds = stratified_shuffle_splits(list(y_all), test_size=args.test_size, repeats=args.repeats, seed=args.seed)
        print(f"[cv] Built {len(folds)} splits ({args.cv_mode})")

    # Registry
    reg = make_registry(seed=args.seed)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in reg:
            raise ValueError(f"Unknown model key: {k}")

    # Run CV
    rows: List[Dict[str, Any]] = []
    for model_key in keys:
        spec = reg[model_key]
        print(f"\n===== MODEL: {model_key} =====")
        for fold_id, (tr, te) in enumerate(folds, 1):
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]

            # Model specific DataFrame views
            if model_key == "brcg":
                dftr = df_all_brcg.iloc[tr, :]
                dfte = df_all_brcg.iloc[te, :]
            elif model_key == "brl_aix360":
                # BRL usually fine with numeric, keep standard DF
                dftr = df_all.iloc[tr, :]
                dfte = df_all.iloc[te, :]
            else:
                dftr = None
                dfte = None

            metrics = eval_one_model_fold(model_key, spec, Xtr, ytr, Xte, yte, dftr, dfte, colnames=colnames)
            status = metrics.pop("_status")
            if status != "ok":
                print(f"[{model_key}][fold {fold_id}] {status} — skipping this fold.")
                continue

            metrics.update({"fold": fold_id})
            rows.append({"model": model_key, **metrics})
            auc_show = metrics['roc_auc'] if not (metrics['roc_auc'] is None or (isinstance(metrics['roc_auc'], float) and math.isnan(metrics['roc_auc']))) else float('nan')
            print(f"[{model_key}][fold {fold_id}] acc={metrics['acc']:.3f} bal={metrics['bal_acc']:.3f} "
                  f"f1M={metrics['f1_macro']:.3f} auc={auc_show:.3f} "
                  f"rules={metrics['n_rules']} body={metrics['avg_body']} "
                  f"tree(nodes/depth)={metrics['tree_nodes']}/{metrics['tree_depth']} "
                  f"time(train/pred)={metrics['time_train']:.2f}/{metrics['time_pred']:.2f}s")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model","fold",
            "acc","bal_acc","f1_macro","roc_auc","pr_auc","brier","ece10",
            "tn","fp","fn","tp",
            "n_rules","avg_body","n_features","tree_nodes","tree_depth",
            "time_train","time_pred","time_total"
        ])
        def fnum(x):
            if x is None: return "nan"
            try:
                if isinstance(x, float) and math.isnan(x):
                    return "nan"
            except Exception:
                pass
            try:
                return f"{x:.6f}"
            except Exception:
                return str(x)
        for r in rows:
            w.writerow([
                r["model"], r["fold"],
                fnum(r['acc']), fnum(r['bal_acc']), fnum(r['f1_macro']),
                fnum(r['roc_auc']), fnum(r['pr_auc']), fnum(r['brier']), fnum(r['ece10']),
                r["tn"], r["fp"], r["fn"], r["tp"],
                fnum(r['n_rules']), fnum(r['avg_body']), fnum(r['n_features']),
                fnum(r['tree_nodes']), fnum(r['tree_depth']),
                fnum(r['time_train']), fnum(r['time_pred']), fnum(r['time_total']),
            ])
    print(f"\n[baselines] wrote per-fold results to {args.out_csv}")


if __name__ == "__main__":
    main()

