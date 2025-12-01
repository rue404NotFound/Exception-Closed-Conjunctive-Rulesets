# ECCRS: Exception Closed Conjunctive Rule Sets

This repository contains the implementation of the ECCRS classifier and the scripts used to evaluate it against baseline models.

## Structure

- `eccrs/`: Core ECCRS implementation (data loading, certificate building, projector, selector, trainer, fallbacks)
- `main.py`: Single split runner for ECCRS with selectors, fallbacks and laminar closure
- `cv.py`: Cross validation runner for ECCRS
- `run_suite.py`: Script to run ECCRS (with different fallback modes) and baselines on the same splits
- `baselines.py`: Python baselines (trees, rule learners, linear models, kNN, etc.) on the same LP datasets
- `baselines.py`: Python baselines on the same LP datasets, including:
                  - Decision trees: `dt_gini`, `dt_entropy`
                  - Rule learners: `ripper`, `brcg`, `brl_imodels`, `grl`, `rulefit`
                  - Ensemble and boosting models: `rf` (RandomForestClassifier), `gbdt` (GradientBoostingClassifier), `hgbdt` (HistGradientBoostingClassifier), `ebm` (ExplainableBoostingClassifier)
                  - Linear and kernel models: `logreg` (LogisticRegression), `svm_linear`, `svm_rbf`
                  - Naive Bayes: `nb_bern` (BernoulliNB), `nb_gauss` (GaussianNB)
