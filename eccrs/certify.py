# eccrs/certify.py
# Certificate-first: build minimal decisive sets for mistakes, with prime reduction.
# Optimized to use Python big-int bitsets (no 64-bit overflow) and a tight greedy loop.

from typing import Tuple, Set, Optional, FrozenSet
import numpy as np
from .data import Dataset

# A literal is (attribute_index, value)
Literal = Tuple[int, int]


def coverage_of_body(ds: Dataset, body: Set[Literal]) -> int:
    """
    Compute the bitset of rows that satisfy all literals in `body`.
    If body is empty, return bitset of all rows.
    """
    if not body:
        return ds.ALL

    cov = ds.ALL
    bitsets = ds.bitsets  # local alias for speed

    for lit in body:
        cov &= bitsets[lit]
        # Early exit: if coverage is already empty, no row satisfies all literals
        if cov == 0:
            return 0

    return cov


def certificate_for_row(ds: Dataset, i: int, y_true: int) -> Optional[FrozenSet[Literal]]:
    """
    Greedy certificate computation for a single row.

    Idea:
    - Consider only rows of the *opposite* class.
    - Build a body S of literals that are true on row i.
    - Repeatedly add the literal that most reduces the number of
      opposite-class rows that still satisfy all literals in S.
    - Stop when no opposite-class row satisfies S (decisive) or when
      no literal can reduce the opposite set further (failure).
    - Prime-reduce S at the end by removing redundant literals.

    Returns:
        - frozenset of literals (minimal decisive body) on success
        - None if no decisive body can be found
    """
    # Row i as a dictionary {attr: value}
    row = ds.rows[i]

    # Candidate literals: those that are true on this row
    # Use an ordered list for deterministic behavior.
    L_list = [(j, v) for j, v in row.items()]
    n_lits = len(L_list)

    if n_lits == 0:
        # No literals to use -> cannot build a certificate
        return None

    # Bitmask of rows with the opposite label to y_true
    opp_mask = ds.Y_neg if y_true == 1 else ds.Y_pos

    # Current body S and its coverage
    S: Set[Literal] = set()
    cov_S = ds.ALL  # coverage of empty set is "all rows"
    opp_remain = opp_mask & cov_S

    # If even the empty body eliminates all opposite rows, we are already decisive
    if opp_remain == 0:
        return frozenset()

    bitsets = ds.bitsets  # local alias
    popcount = int.bit_count  # unbound method used as a function

    # Precompute bitsets of each candidate literal, restricted to opposite-class rows.
    # This avoids intersecting with opp_mask inside the greedy loop.
    L_bits_opp = [bitsets[lit] & opp_mask for lit in L_list]

    # Track which literals have been used
    used = np.zeros(n_lits, dtype=bool)

    # Greedy loop: keep adding literals until no opposite row remains or no progress is possible
    while opp_remain != 0:
        current_size = popcount(opp_remain)

        # Initialize all counts to "worse than current" so used literals
        # automatically lose the argmin competition.
        remaining_counts = np.full(n_lits, current_size + 1, dtype=np.int32)

        # Compute, for each unused literal, how many opposite rows would remain
        # if we add it to the current body.
        for idx in range(n_lits):
            if used[idx]:
                continue

            candidate_mask = opp_remain & L_bits_opp[idx]
            remaining_counts[idx] = popcount(candidate_mask)

        # Choose the literal that minimizes the remaining opposite rows
        best_idx = int(np.argmin(remaining_counts))
        best_size = int(remaining_counts[best_idx])

        # No progress: all unused literals leave at least as many opposite rows
        if best_size >= current_size:
            return None

        # Add the best literal to S
        best_lit = L_list[best_idx]
        S.add(best_lit)
        used[best_idx] = True

        # Update coverage and remaining opposite rows
        cov_S &= bitsets[best_lit]
        opp_remain = opp_mask & cov_S

        # If we managed to eliminate all opposite rows, we are done with the greedy phase
        if best_size == 0 or opp_remain == 0:
            break

    # If we still have opposite rows, then no decisive body was found
    if opp_remain != 0:
        return None

    # Prime reduction: remove literals that are not necessary.
    # Try dropping each literal and see if S still excludes all opposite-class rows.
    for lit in list(S):
        trial = set(S)
        trial.remove(lit)
        cov_trial = coverage_of_body(ds, trial)
        if (opp_mask & cov_trial) == 0:
            # Removing this literal keeps the body decisive; drop it permanently.
            S.remove(lit)

    return frozenset(S)
