
import subprocess, sys, json
from pathlib import Path
from datetime import datetime
import argparse

def run_cmd(argv, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(">>", " ".join(map(str, argv)))
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, shell=False)
    if p.returncode != 0:
        print(f"[ERR] command failed: {log_path}")
        sys.exit(p.returncode)

def dataset_slug(path: str) -> str:
    p = Path(path)
    stem = p.stem
    if stem.startswith("data-"):
        stem = stem[5:]
    return stem

def common_cv_args(args):
    a = [
        sys.executable, "cv.py",
        "--data", args.data,
        "--label_attr", str(args.label_attr),
        "--cv_mode", args.cv_mode,
        "--test_size", str(args.test_size),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--max_iters", str(args.max_iters),
        "--selector", args.selector,
        "--mdl_c0", str(args.mdl_c0),
        "--mdl_c1", str(args.mdl_c1),
        "--mdl_eta", str(args.mdl_eta),
    ]
    if args.use_selected:           a += ["--use_selected"]
    if args.laminar_strict:         a += ["--laminar_strict", "--laminar_min_overlap", str(args.laminar_min_overlap)]
    return a

def run_eccrs_first_with_folds(args, OUTROOT, folds_file, fallback_key, out_tag, runstamp):
    out_csv  = OUTROOT / f"results_{out_tag}_eccrs_{args.cv_mode}{int(args.test_size*100)}_{args.repeats}x_seed{args.seed}_{fallback_key}.csv"
    rulesdir = OUTROOT / f"rules_{fallback_key}"
    log      = OUTROOT / f"log_{fallback_key}_{runstamp}.txt"

    argv = common_cv_args(args) + [
        "--fallback", fallback_key,
        "--save_rules_dir", str(rulesdir),
        "--out_csv", str(out_csv),
        "--save_folds", str(folds_file),
    ]
    if fallback_key in ("knn", "nearest"):
        argv += ["--k", str(args.k), "--feat_weight", args.feat_weight]
        if args.knn_weighted:
            argv.append("--knn_weighted")

    run_cmd(argv, log)
    return str(out_csv), str(rulesdir)

def run_eccrs_on_saved_folds(args, OUTROOT, folds_file, fallback_key, out_tag, runstamp):
    out_csv  = OUTROOT / f"results_{out_tag}_eccrs_{args.cv_mode}{int(args.test_size*100)}_{args.repeats}x_seed{args.seed}_{fallback_key}.csv"
    rulesdir = OUTROOT / f"rules_{fallback_key}"
    log      = OUTROOT / f"log_{fallback_key}_{runstamp}.txt"

    argv = common_cv_args(args) + [
        "--load_folds", str(folds_file),
        "--fallback", fallback_key,
        "--save_rules_dir", str(rulesdir),
        "--out_csv", str(out_csv),
    ]
    if fallback_key in ("knn", "nearest"):
        argv += ["--k", str(args.k), "--feat_weight", args.feat_weight]
        if args.knn_weighted:
            argv.append("--knn_weighted")

    run_cmd(argv, log)
    return str(out_csv), str(rulesdir)

def run_baselines(args, OUTROOT, folds_file, out_tag, runstamp):
    out_csv = OUTROOT / f"results_{out_tag}_baselines_{args.cv_mode}{int(args.test_size*100)}_{args.repeats}x_seed{args.seed}.csv"
    log     = OUTROOT / f"log_baselines_{runstamp}.txt"
    argv = [
        sys.executable, "baselines.py",
        "--data", args.data,
        "--label_attr", str(args.label_attr),
        "--load_folds", str(folds_file),
        "--models", args.baselines_models,
        "--out_csv", str(out_csv),
        "--seed", str(args.seed),
    ]
    run_cmd(argv, log)
    return str(out_csv)

def main():
    ap = argparse.ArgumentParser(description="Run ECCRS (3 fallbacks) + baselines on same folds.")
    # Data
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--label_attr", type=int, required=True)
    # CV
    ap.add_argument("--cv_mode", type=str, choices=["kfold","shuffle"], default="shuffle")
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--test_size", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=8)
    # ECCRS
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--selector", type=str, choices=["none","mdl","fallback_gain"], default="mdl")
    ap.add_argument("--mdl_c0", type=float, default=50)
    ap.add_argument("--mdl_c1", type=float, default=10)
    ap.add_argument("--mdl_eta", type=float, default=0.5)
    ap.add_argument("--use_selected", action="store_true", default=True)
    # Fallback params
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--knn_weighted", action="store_true", default=True)
    ap.add_argument("--feat_weight", type=str, choices=["none","mi"], default="mi")
    # Laminar strict
    ap.add_argument("--laminar_strict", action="store_true", default=True)
    ap.add_argument("--laminar_min_overlap", type=int, default=0)
    # Baselines
    ap.add_argument("--baselines_models", type=str,
                    default="dt_gini,dt_entropy,ripper,brl,brcg,logreg,knn")
    # Output root and optional tag
    ap.add_argument("--out_root", type=str, default="experiments")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    slug = dataset_slug(args.data)
    out_tag = slug if not args.tag else f"{slug}_{args.tag}"
    RUNSTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTROOT = Path(args.out_root) / slug
    OUTROOT.mkdir(parents=True, exist_ok=True)

    # folds file
    if args.cv_mode == "kfold":
        folds_file = OUTROOT / f"folds_{slug}_k{args.kfolds}_{args.repeats}x_seed{args.seed}.json"
    else:
        folds_file = OUTROOT / f"folds_{slug}_{args.cv_mode}{int(args.test_size*100)}_{args.repeats}x_seed{args.seed}.json"

    manifest = {
        "dataset": str(Path(args.data).resolve()),
        "label_attr": args.label_attr,
        "cv": {
            "mode": args.cv_mode, "kfolds": args.kfolds, "test_size": args.test_size,
            "repeats": args.repeats, "seed": args.seed,
            "folds_file": str(folds_file.resolve()),
        },
        "eccrs": {},
        "baselines_csv": None,
        "timestamp": RUNSTAMP,
    }

    # 1) First ECCRS run writes folds: use knn by default (can be any)
    knn_csv, knn_rules = run_eccrs_first_with_folds(args, OUTROOT, folds_file, "knn", out_tag, RUNSTAMP)
    manifest["eccrs"]["knn"] = {"csv": knn_csv, "rules_dir": knn_rules}

    # 2) Re-use folds for 'nearest' and 'abst'
    near_csv, near_rules = run_eccrs_on_saved_folds(args, OUTROOT, folds_file, "nearest", out_tag, RUNSTAMP)
    manifest["eccrs"]["nearest"] = {"csv": near_csv, "rules_dir": near_rules}

    abst_csv, abst_rules = run_eccrs_on_saved_folds(args, OUTROOT, folds_file, "abst", out_tag, RUNSTAMP)
    manifest["eccrs"]["abst"] = {"csv": abst_csv, "rules_dir": abst_rules}

    # 3) Baselines on same folds
    bl_csv = run_baselines(args, OUTROOT, folds_file, out_tag, RUNSTAMP)
    manifest["baselines_csv"] = bl_csv

    man_path = OUTROOT / f"manifest_{out_tag}.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n[OK] All done. Manifest:\n ", man_path.resolve())
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()

