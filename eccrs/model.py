# eccrs/model.py
# ECCRS rule structure and inference (most-specific-wins) with ABST default.

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from .data import Dataset

Literal = Tuple[int, int]  # (attr, val) where val in {0,1}


@dataclass
class Rule:
    body: frozenset  # frozenset[Literal]
    label: int  # 0 or 1
    cov_bits: int  # coverage bitset over training data (where created)
    witness: Optional[int] = None  # training row index used to create it


@dataclass
class RuleSet:
    rules: List[Rule] = field(default_factory=list)

    def applicable_rules(self, row_literals: Set[Literal]) -> List[Rule]:
        return [r for r in self.rules if r.body.issubset(row_literals)]

    def predict_row(self, row_literals: Set[Literal]) -> Optional[int]:
        """
        ECCRS decision: if multiple applicable rules, undominated maxima by inclusion survive.
        If survivors share the same label => return it; else abstain (⊥).

        Under ECCRS exception-closure this yields a single label on covered points.
        """
        applicable = [r for r in self.rules if r.body.issubset(row_literals)]
        if not applicable:
            return None  # ABST

        maxima: List[Rule] = []
        for r in applicable:
            # Remove any existing maxima that r strictly includes
            maxima = [
                m
                for m in maxima
                if not (m.body.issubset(r.body) and len(m.body) < len(r.body))
            ]
            # If r is strictly included by any current maximum, skip it
            if any(
                r.body.issubset(m.body) and len(r.body) < len(m.body)
                for m in maxima
            ):
                continue
            maxima.append(r)

        labels = {m.label for m in maxima}
        if len(labels) == 1:
            return next(iter(labels))
        return None  # Tie (should not happen on covered points if ECCRS holds)

    def predict_all(self, ds: Dataset) -> List[Optional[int]]:
        preds: List[Optional[int]] = []
        # Reuse cached literals per row
        for lits in ds.row_literals:
            preds.append(self.predict_row(lits))
        return preds
