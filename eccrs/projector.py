# eccrs/projector.py
# Laminar projector + prime reduction; keeps ECCRS constraints intact.
# OPTIMIZED: Filter opposite-label rules before checking overlap

from typing import Tuple, Set, Optional
from .data import Dataset
from .model import Rule, RuleSet

Literal = Tuple[int, int]

def popcount(x: int) -> int:
    return x.bit_count()

def coverage_of_body(ds: Dataset, body: Set[Literal]) -> int:
    if not body:
        return ds.ALL
    cov = ds.ALL
    for (j, v) in body:
        cov &= ds.bitsets[(j, v)]
    return cov

def overlap_bits(ds: Dataset, body_a: Set[Literal], body_b: Set[Literal]) -> int:
    return coverage_of_body(ds, body_a) & coverage_of_body(ds, body_b)

def comparable(a: Set[Literal], b: Set[Literal]) -> bool:
    return a.issubset(b) or b.issubset(a)

def is_decisive(ds: Dataset, body: Set[Literal], label: int) -> bool:
    """No opposite-class training row satisfies 'body'."""
    opp = ds.Y_neg if label == 1 else ds.Y_pos
    return (coverage_of_body(ds, body) & opp) == 0

def covers_witness(ds: Dataset, body: Set[Literal], witness: int) -> bool:
    row = ds.rows[witness]
    for (j, v) in body:
        if row.get(j) != v:
            return False
    return True

def specialize_to_break_overlap(
    ds: Dataset,
    new_body: Set[Literal],
    witness_lits: Set[Literal],
    other_body: Set[Literal]
) -> Optional[frozenset]:
    """
    Try to add literals (from witness_lits) to new_body to remove overlap with other_body,
    while still covering the witness. If impossible, enforce comparability by ensuring
    other_body ⊂ new_body (add other_body's literals that the witness also has).
    """
    cur = set(new_body)
    ov = overlap_bits(ds, cur, other_body)
    
    # Phase 1: attempt to eliminate overlap
    while ov != 0:
        best_lit = None
        best_size = popcount(ov) + 1
        for lit in (witness_lits - cur):
            s = popcount(ov & ds.bitsets[lit])
            if s < best_size:
                best_size = s
                best_lit = lit
                if s == 0:
                    break
        
        if best_lit is None:
            break
        
        cur.add(best_lit)
        ov = overlap_bits(ds, cur, other_body)
    
    if ov == 0:
        return frozenset(cur)
    
    # Phase 2: force comparability if possible
    to_add = [lit for lit in other_body if lit in witness_lits and lit not in cur]
    if to_add:
        cur.update(to_add)
        return frozenset(cur)
    
    return None

def prime_reduce_preserving_semantics(ds: Dataset, rs: RuleSet, r: Rule) -> frozenset:
    """
    Reduce r.body by removing literals where possible, but ALWAYS preserve:
    - witness coverage,
    - decisiveness against the opposite class,
    - laminarity (no flat cross-class overlaps: any overlap must be comparable).
    Never return an empty body.
    """
    body = set(r.body)
    
    def ok(trial: Set[Literal]) -> bool:
        if not trial:
            return False
        if not covers_witness(ds, trial, r.witness):
            return False
        if not is_decisive(ds, trial, r.label):
            return False
        
        # laminarity vs opposite-class rules
        for q in rs.rules:
            if q.label == r.label:
                continue
            ov = overlap_bits(ds, trial, set(q.body))
            if ov != 0 and not comparable(trial, set(q.body)):
                return False
        return True
    
    # Greedy try dropping literals
    changed = True
    while changed:
        changed = False
        for lit in list(body):
            trial = set(body)
            trial.remove(lit)
            if ok(trial):
                body = trial
                changed = True
    
    return frozenset(body)

def projector_add(
    ds: Dataset,
    rs: RuleSet,
    candidate_body: frozenset,
    label: int,
    witness: int
) -> Optional[Rule]:
    """
    OPTIMIZED: Filter opposite-label rules to only those that can overlap on training data.
    
    ECCRS-safe add:
    1) Ensure witness coverage and decisiveness for the candidate.
    2) Resolve flat overlaps with opposite-class rules by specializing from witness.
    3) Prime-reduce while preserving witness, decisiveness, and laminarity.
    4) Dedup AFTER reduction. Reject empty-body (⊤) rules during training.
    """
    if not candidate_body:
        return None
    
    new_body = set(candidate_body)
    
    if not covers_witness(ds, new_body, witness):
        return None
    if not is_decisive(ds, new_body, label):
        return None
    
    witness_lits = {(j, ds.rows[witness][j]) for j in ds.features if j in ds.rows[witness]}
    
    # OPTIMIZATION: Pre-compute candidate coverage ONCE
    cand_cov = coverage_of_body(ds, new_body)
    
    # OPTIMIZATION: Filter to opposite-label rules that CAN overlap on training data
    # This is the key speedup: bitset intersection is O(1), avoids checking hundreds of rules
    opp_rules = [
        q for q in rs.rules 
        if q.label != label and (q.cov_bits & cand_cov) != 0
    ]
    
    # Now only check the filtered subset (typically 5-20% of total rules)
    for q in opp_rules:
        q_body = set(q.body)
        
        # Check if already comparable (fast set operations)
        if new_body.issubset(q_body) or q_body.issubset(new_body):
            continue
        
        # Not comparable and overlaps: must specialize
        specialized = specialize_to_break_overlap(ds, new_body, witness_lits, q_body)
        if specialized is None:
            return None
        
        new_body = set(specialized)
        cand_cov = coverage_of_body(ds, new_body)  # Recompute after change
        
        if not covers_witness(ds, new_body, witness):
            return None
        if not is_decisive(ds, new_body, label):
            return None
    
    # Prime reduce
    tmp = Rule(body=frozenset(new_body), label=label, cov_bits=0, witness=witness)
    reduced = prime_reduce_preserving_semantics(ds, rs, tmp)
    
    if not reduced:
        return None
    
    # Dedup AFTER reduction (and unify label by majority on coverage)
    cov = coverage_of_body(ds, set(reduced))
    
    # If identical body exists, unify by majority on coverage
    same = [r for r in rs.rules if r.body == reduced]
    if same:
        pos = popcount(cov & ds.Y_pos)
        neg = popcount(cov & ds.Y_neg)
        keep_label = 1 if pos >= neg else 0
        rs.rules = [r for r in rs.rules if r.body != reduced]
        new_rule = Rule(body=reduced, label=keep_label, cov_bits=cov, witness=witness)
        rs.rules.append(new_rule)
        return new_rule
    
    new_rule = Rule(body=reduced, label=label, cov_bits=cov, witness=witness)
    rs.rules.append(new_rule)
    return new_rule




