# if __name__ == "__main__":
#     main()

import argparse
import random
from eccrs.data import load_lp, Dataset
from eccrs.trainer import (
    train_cf_eccrs, audit_eccrs_global, evaluate_with_fallback,
    run_mdl_selector, run_fallback_gain_selector, print_ruleset,
    prune_redundant_same_label, laminar_strict_closure, ensure_at_least_one_pair,
)

def main():
    ap = argparse.ArgumentParser(description="CF-ECCRS with selectors, fallbacks, and laminar options.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--label_attr", type=int, default=10)
    ap.add_argument("--ignore_attr", type=int, action="append", default=[])

    ap.add_argument("--split", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iters", type=int, default=5000)

    # Fallback controls
    ap.add_argument("--fallback", type=str, default="abst", choices=["abst", "nearest", "knn"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--knn_weighted", action="store_true")
    ap.add_argument("--feat_weight", type=str, default="none", choices=["none", "mi"])

    # Training gates
    ap.add_argument("--max_rules", type=int, default=10**9)
    ap.add_argument("--min_cov_pos", type=int, default=1)
    ap.add_argument("--min_gain", type=int, default=1)

    # Selectors
    ap.add_argument("--selector", type=str, default="none", choices=["none", "mdl", "fallback_gain"])
    ap.add_argument("--mdl_c0", type=float, default=2.0)
    ap.add_argument("--mdl_c1", type=float, default=1.0)
    ap.add_argument("--mdl_eta", type=float, default=2.0)
    ap.add_argument("--fg_max_rules", type=int, default=100)
    ap.add_argument("--fg_min_delta", type=float, default=0.0)

    # Selection usage
    ap.add_argument("--use_selected", action="store_true")

    # Global strict closure + knobs
    ap.add_argument("--laminar_strict", action="store_true", help="Synthesize exception guards so no opposite-label flat conflicts remain anywhere.")
    ap.add_argument("--laminar_min_overlap", type=int, default=0, help="Only materialize exceptions for pairs whose training overlap has at least this many rows (0 = strict global).")
    ap.add_argument("--laminar_no_overlap_majority", action="store_true", help="Disable majority-on-overlap orientation (defaults to ON).")

    args = ap.parse_args()

    # --- Load data
    ignore_set = set(args.ignore_attr) if args.ignore_attr else set()
    print(f"[load] file={args.data}, label_attr=a({args.label_attr}), ignore={sorted(ignore_set)}")
    ds_full: Dataset = load_lp(args.data, label_attr=args.label_attr, ignore_attrs=ignore_set)
    print(f"[data] rows={ds_full.n}, features={len(ds_full.features)} -> {ds_full.features}")

    # --- Split
    if args.split > 0.0:
        rnd = random.Random(args.seed)
        idxs = list(range(ds_full.n))
        rnd.shuffle(idxs)
        cut = int(len(idxs) * args.split)
        train_idx = sorted(idxs[:cut])
        test_idx = sorted(idxs[cut:])

        def subset(ds: Dataset, keep_idx):
            rows = [ds.rows[i] for i in keep_idx]
            labels = [ds.labels[i] for i in keep_idx]
            sub = Dataset(rows=rows, labels=labels, features=ds.features, label_attr=ds.label_attr)
            if hasattr(ds, "feat_names"):
                try:
                    sub.feat_names = ds.feat_names
                except Exception:
                    pass
            return sub

        ds_train = subset(ds_full, train_idx)
        ds_test = subset(ds_full, test_idx)
        print(f"[split] train={ds_train.n} test={ds_test.n}")
    else:
        ds_train = ds_full
        ds_test = None

    # --- Train
    rs = train_cf_eccrs(
        ds_train,
        max_iters=args.max_iters,
        verbose=True,
        max_rules=args.max_rules,
        min_cov_pos=args.min_cov_pos,
        min_gain=args.min_gain,
    )

    # --- Prune redundant (same-label)
    before = len(rs.rules)
    rs = prune_redundant_same_label(ds_train, rs)
    after = len(rs.rules)
    if after != before:
        print(f"[prune] removed {before - after} redundant same-label rules; K={after}")

    # --- Audit ECCRS on train support overlaps
    ok, msg = audit_eccrs_global(ds_train, rs)
    print(f"[audit] ECCRS constraints (global/train): {'OK' if ok else 'VIOLATION'}{'' if ok else ' - ' + msg}")

    # --- Evaluate before selection
    _ = evaluate_with_fallback(
        rs, ds_train, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="train",
        knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
    )
    if ds_test is not None:
        _ = evaluate_with_fallback(
            rs, ds_test, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="test",
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
        )
    _ = evaluate_with_fallback(
        rs, ds_full, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="final",
        knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
    )
    print_ruleset(rs, ds_full, title="[model] Learned ECCRS rules (body => label):")

    # --- Selection
    rs_sel = None
    if args.selector == "mdl":
        rs_sel = run_mdl_selector(rs, ds_train, c0=args.mdl_c0, c1=args.mdl_c1, eta=args.mdl_eta, verbose=True)
    elif args.selector == "fallback_gain":
        rs_sel = run_fallback_gain_selector(
            rs, ds_train, fallback=args.fallback, k=args.k, train_ref=ds_train,
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
            max_add=args.fg_max_rules, min_delta=args.fg_min_delta, verbose=True,
        )

    if rs_sel is not None:
        _ = evaluate_with_fallback(
            rs_sel, ds_train, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="train_selected",
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
        )
        if ds_test is not None:
            _ = evaluate_with_fallback(
                rs_sel, ds_test, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="test_selected",
                knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
            )
        print_ruleset(rs_sel, ds_full, title="[model_selected] Selected rules (body => label):")
        if args.use_selected:
            rs = rs_sel
            print("[use_selected] Using selected subset for final snapshot.")

    # --- ALWAYS ensure at least one explicit default→exception pair
    rs = ensure_at_least_one_pair(rs, ds_train, verbose=True)

    # --- Optional: strict global closure after the pair is present
    if args.laminar_strict:
        print("[laminar_strict] Enforcing global default→exception closure...")
        rs = laminar_strict_closure(
            rs, ds_train, verbose=True,
            laminar_min_overlap=args.laminar_min_overlap,
            orient_by_overlap=(not args.laminar_no_overlap_majority),
        )
        _ = evaluate_with_fallback(
            rs, ds_full, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="final_after_strict",
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
        )
        if ds_test is not None:
            _ = evaluate_with_fallback(
                rs, ds_test, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="test_after_strict",
                knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
            )
        _ = evaluate_with_fallback(
            rs, ds_full, fallback=args.fallback, k=args.k, train_ref=ds_train, tag="final_after_strict",
            knn_weighted=args.knn_weighted, feat_weight=args.feat_weight,
        )
    print_ruleset(rs, ds_full, title="[final_model] Rules in use (body => label):")

if __name__ == "__main__":
    main()
