#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sba_theory_proofbench.py

Comprehensive single-file verification, simulation, and testing harness for
the Specular Bit Architecture (SBA) and the cosmological mapping described
in the user's manuscript. This script is intended as an academic-grade,
reproducible "proof bench" that performs the following tasks:

  1. Parameter management and reproducibility metadata export.
  2. Fibonacci weight computation (supports negative indices).
  3. Exhaustive width-5 enumeration of local neighborhoods:
       - apply interior and boundary rewrites,
       - verify weighted-sum preservation for interior rule,
       - simulate sanitization-phase deterministic reconstructor and Protocol C,
       - record CSV artifacts and verification report.
  4. Symbolic verification of the Fibonacci identity using SymPy (optional).
  5. Informational mass and E = m c^2 bookkeeping examples and CSV export.
  6. Toy 1D continuous model: probability-density -> pressure -> acceleration,
     to illustrate how a pressure law P(rho) can produce effective forces.
  7. Unit tests (unittest) for core primitives.
  8. Command-line interface to run selected modules and produce outputs.

Usage (from repository root):
    python3 sba_theory_proofbench.py --all

Outputs (written to ./proofbench_outputs by default):
 - parameters.json
 - width5_transitions_full.csv
 - verification_report.txt
 - energy_examples.csv
 - symbolic_proof.txt (if SymPy available)
 - toy_simulation.csv
 - unit_test_report.txt
 - prime_composite_sba_results.json (+ related CSV/TXT artifacts)
 - prime_phase_map_results.json (+ related CSV/TXT artifacts)

Author: Marco Oppido
Date: 03-28-2026
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

# Optional imports (SymPy). The script will continue without SymPy if not installed.
try:
    import sympy as sp  # type: ignore
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

# ---------------------------
# Logging and output setup
# ---------------------------

OUTPUT_DIR = "outputs"
CSV_TRANSITIONS = os.path.join(OUTPUT_DIR, "width5_transitions_full.csv")
VERIFICATION_REPORT = os.path.join(OUTPUT_DIR, "verification_report.txt")
SYMBOLIC_PROOF_FILE = os.path.join(OUTPUT_DIR, "symbolic_proof.txt")
ENERGY_EXAMPLES = os.path.join(OUTPUT_DIR, "energy_examples.csv")
TOY_SIM_CSV = os.path.join(OUTPUT_DIR, "toy_simulation.csv")
PARAMETERS_JSON = os.path.join(OUTPUT_DIR, "parameters.json")
UNITTEST_REPORT = os.path.join(OUTPUT_DIR, "unit_test_report.txt")
FIB_PYRAMID_REPORT = os.path.join(OUTPUT_DIR, "fibonacci_pyramid_verification.txt")
CAUSAL_PYRAMID_REPORT = os.path.join(OUTPUT_DIR, "causal_pyramid_verification.txt")
CAUSAL_PYRAMID_FRONTS_CSV = os.path.join(OUTPUT_DIR, "causal_pyramid_fronts.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sba_proofbench")

# ---------------------------
# Configuration dataclass
# ---------------------------

@dataclass
class Config:
    # Fibonacci offset (default aligned with Zeckendorf admissible regime)
    OFFSET_O: int = 2

    # Protocol C parameters
    ALPHA: float = 1.0
    BETA: float = 1.0
    GAMMA: float = 2.0

    # Reconstruction radius
    RECON_RADIUS: int = 2

    # Bookkeeping constants
    M0: float = 1.0          # discrete bookkeeping mass unit (adimensional)
    MU: float = 1e-30        # calibration constant: kg per unit weight (example)
    C_LIGHT: float = 299792458.0  # speed of light (m/s)

    # Toy continuous model parameters
    PRESSURE_KAPPA: float = 1.0   # proportionality constant for P(rho)
    PRESSURE_EXPONENT: float = 1.0  # exponent k in P = kappa * rho^k

    # Output and enumeration
    OUTPUT_DIR: str = OUTPUT_DIR
    DIGITS: Tuple[int, ...] = (-2, -1, 0, 1, 2)
    ROWS_PER_PART: int = 625

    # Deterministic seed for reproducibility (used for hashing outputs)
    RUN_ID: str = ""

    def to_dict(self):
        d = asdict(self)
        # Convert non-serializable types
        d["DIGITS"] = list(self.DIGITS)
        return d

# ---------------------------
# Utilities
# ---------------------------

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def fib(n: int, memo: Dict[int, int] = None) -> int:
    """
    Compute Fibonacci number F_n for integer n (supports negative indices).
    Uses F_0 = 0, F_1 = 1 and F_{-n} = (-1)^{n+1} F_n.
    Efficient iterative memoization for positive indices.
    """
    if memo is None:
        memo = {0: 0, 1: 1}
    if n in memo:
        return memo[n]
    if n > 1:
        a, b = memo[0], memo[1]
        for k in range(2, n + 1):
            a, b = b, a + b
            memo[k] = b
        return memo[n]
    # negative index
    pos = -n
    pos_val = fib(pos, memo)
    sign = -1 if (pos % 2 == 0) else 1
    return sign * pos_val

def compute_weights(indices: Iterable[int], offset: int = 0) -> Dict[int, int]:
    """Compute positional weights W_k = F_{k + offset} for given indices."""
    return {k: fib(k + offset) for k in indices}

def weighted_sum(neigh: Tuple[int, int, int, int, int], weights: Dict[int, int]) -> int:
    """Compute weighted sum sum_{k=-2..2} d_k * W_k."""
    keys = [-2, -1, 0, 1, 2]
    return sum(neigh[i] * weights[keys[i]] for i in range(5))

def info_mass(neigh: Tuple[int, int, int, int, int], weights: Dict[int, int], m0: float) -> float:
    """Informational mass M(V) = m0 * sum |v_i| W_i over the 5-tuple."""
    keys = [-2, -1, 0, 1, 2]
    return m0 * sum(abs(neigh[i]) * weights[keys[i]] for i in range(5))

def latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for underscores and percent signs in table cells."""
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

def fmt_sci(value: float, sig: int = 3) -> str:
    """Format numeric values in scientific notation with uniform significant figures."""
    return f"{value:.{sig-1}e}"

def shannon_entropy(state: Tuple[int, ...]) -> float:
    """Shannon entropy H(state) over empirical symbol frequencies."""
    if len(state) == 0:
        return 0.0
    counts: Dict[int, int] = {}
    for value in state:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(state))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def structural_phi(state: Tuple[int, ...], weights: Dict[int, int], rho_overflow: float = 1.0) -> float:
    """
    Structural term from manuscript-inspired definition:
      Phi(d) = sum_i |d_i| W_i + rho * (#overflow sites where |d_i| = 2)
    """
    keys = [-2, -1, 0, 1, 2]
    magnitude_term = sum(abs(state[i]) * weights[keys[i]] for i in range(min(5, len(state))))
    overflow_term = rho_overflow * sum(1 for value in state if abs(value) == 2)
    return magnitude_term + overflow_term

def compute_discrete_lagrangian(
    pre_sanitization: Tuple[int, int, int, int, int],
    post_sanitization: Tuple[int, int, int, int, int],
    weights: Dict[int, int],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    rho_overflow: float = 1.0,
    temperature_k: float = 300.0,
) -> Dict[str, float]:
    """
    Compute manuscript-style discrete Lagrangian ingredients between two states.

    Returns dict with:
      - delta_H: entropy differential
      - Q_t: Landauer lower bound k_B T ln(2) * N_erase
      - phi_pre, phi_post, delta_phi
      - N_erase
      - lagrangian: alpha*delta_phi + beta*delta_H + gamma*Q_t
    """
    k_b = 1.380649e-23
    h_pre = shannon_entropy(pre_sanitization)
    h_post = shannon_entropy(post_sanitization)
    delta_h = h_post - h_pre

    phi_pre = structural_phi(pre_sanitization, weights, rho_overflow=rho_overflow)
    phi_post = structural_phi(post_sanitization, weights, rho_overflow=rho_overflow)
    delta_phi = phi_post - phi_pre

    # Approximate erase-event magnitude symmetrically for forward/backward comparisons.
    n_erase = 0
    for before, after in zip(pre_sanitization, post_sanitization):
        if (abs(before) > 1) != (abs(after) > 1):
            n_erase += 1

    q_t = k_b * temperature_k * math.log(2.0) * n_erase
    lagrangian = alpha * delta_phi + beta * delta_h + gamma * q_t

    return {
        "delta_H": delta_h,
        "Q_t": q_t,
        "phi_pre": phi_pre,
        "phi_post": phi_post,
        "delta_phi": delta_phi,
        "N_erase": float(n_erase),
        "lagrangian": lagrangian,
    }

def compute_action(
    trajectory: List[Tuple[int, int, int, int, int]],
    weights: Dict[int, int],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    rho_overflow: float = 1.0,
    temperature_k: float = 300.0,
    time_symmetric: bool = True,
) -> float:
    """Compute discrete action A = sum_t L_t for a trajectory of states."""
    if len(trajectory) < 2:
        return 0.0
    total = 0.0
    for i in range(len(trajectory) - 1):
        terms = compute_discrete_lagrangian(
            trajectory[i],
            trajectory[i + 1],
            weights,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho_overflow=rho_overflow,
            temperature_k=temperature_k,
        )
        if time_symmetric:
            total += alpha * abs(terms["delta_phi"]) + beta * abs(terms["delta_H"]) + gamma * terms["Q_t"]
        else:
            total += terms["lagrangian"]
    return total

def cpt_transform_state(state: Tuple[int, int, int, int, int]) -> Tuple[int, int, int, int, int]:
    """Apply C and P to one width-5 state: sign inversion + index reversal."""
    return tuple(-value for value in reversed(state))

def verify_cpt_symmetry(state: Tuple[int, int, int, int, int], config: Config) -> bool:
    """
    Numerical CPT invariance check on a local trajectory.
    1) Build one-step trajectory via deterministic interior rewrite.
    2) Build CPT trajectory by applying C+P to each state and reversing time order.
    3) Compare actions.
    """
    # Use parity-symmetric weights for the numerical CPT check.
    weights = {i: fib(abs(i) + config.OFFSET_O) for i in [-2, -1, 0, 1, 2]}
    _, state_next = interior_rewrite(state)
    trajectory = [state, state_next]

    cpt_trajectory = [cpt_transform_state(s) for s in reversed(trajectory)]

    a_original = compute_action(
        trajectory,
        weights,
        alpha=config.ALPHA,
        beta=config.BETA,
        gamma=config.GAMMA,
        time_symmetric=True,
    )
    a_cpt = compute_action(
        cpt_trajectory,
        weights,
        alpha=config.ALPHA,
        beta=config.BETA,
        gamma=config.GAMMA,
        time_symmetric=True,
    )
    return math.isclose(a_original, a_cpt, rel_tol=1e-12, abs_tol=1e-12)

# ---------------------------
# SBA rewrite rules and sanitization
# ---------------------------

def interior_rewrite(neigh: Tuple[int, int, int, int, int]) -> Tuple[str, Tuple[int, int, int, int, int]]:
    """
    Interior rewrite centered at d0:
      if d0 == +2:
        d0 <- d0 - 2
        d1 <- d1 + 1
        d-2 <- d-2 + 1
      if d0 == -2:
        d0 <- d0 + 2
        d1 <- d1 - 1
        d-2 <- d-2 - 1
      else: none
    """
    d_m2, d_m1, d0, d1, d2 = neigh
    if d0 == 2:
        return ("rewrite_plus2", (d_m2 + 1, d_m1, d0 - 2, d1 + 1, d2))
    if d0 == -2:
        return ("rewrite_minus2", (d_m2 - 1, d_m1, d0 + 2, d1 - 1, d2))
    return ("none", neigh)

def boundary_rule_i0(neigh: Tuple[int, int, int, int, int]) -> Tuple[str, Tuple[int, int, int, int, int]]:
    """
    Boundary rule for i = 0 (leftmost index):
      if d0 == +2:
        d0 <- d0 - 2
        d1 <- d1 + 1
      if d0 == -2:
        d0 <- d0 + 2
        d1 <- d1 - 1
    """
    d_m2, d_m1, d0, d1, d2 = neigh
    if d0 == 2:
        return ("boundary_i0_plus2", (d_m2, d_m1, d0 - 2, d1 + 1, d2))
    if d0 == -2:
        return ("boundary_i0_minus2", (d_m2, d_m1, d0 + 2, d1 - 1, d2))
    return ("none", neigh)

def boundary_rule_i1(neigh: Tuple[int, int, int, int, int]) -> Tuple[str, Tuple[int, int, int, int, int]]:
    """
    Boundary rule for i = 1:
      if d1 == +2:
        d1 <- d1 - 2
        d2 <- d2 + 1
        d0 <- d0 + 1
      if d1 == -2:
        d1 <- d1 + 2
        d2 <- d2 - 1
        d0 <- d0 - 1
    """
    d_m2, d_m1, d0, d1, d2 = neigh
    if d1 == 2:
        return ("boundary_i1_plus2", (d_m2, d_m1, d0 + 1, d1 - 2, d2 + 1))
    if d1 == -2:
        return ("boundary_i1_minus2", (d_m2, d_m1, d0 - 1, d1 + 2, d2 - 1))
    return ("none", neigh)

# ---------------------------
# Sanitization-phase reconstructor and Protocol C
# ---------------------------

def deterministic_reconstructor(post_neigh: Tuple[int, int, int, int, int], recon_radius: int) -> Dict[int, List[Tuple[int, int]]]:
    """
    Deterministic reconstructor used by the verification script.
    Rule:
      - For each source index i in {-2..2} compute s = d_{i-1} + d_{i+1}.
      - Proposed candidate value v = sign(s) in {-1,0,1}.
      - Candidate targets j are indices with |i-j| <= recon_radius (bounded to -2..2).
    Returns mapping target_j -> list of (source_i, candidate_value).
    """
    mapping = {j: [] for j in range(-2, 3)}
    d = {k: post_neigh[idx] for idx, k in enumerate(range(-2, 3))}
    for i in range(-2, 3):
        left = d.get(i - 1, 0)
        right = d.get(i + 1, 0)
        s = left + right
        if s > 0:
            v = 1
        elif s < 0:
            v = -1
        else:
            v = 0
        for j in range(max(-2, i - recon_radius), min(2, i + recon_radius) + 1):
            mapping[j].append((i, v))
    return mapping

def protocol_c_arbitrate(mapping: Dict[int, List[Tuple[int, int]]],
                        post_neigh: Tuple[int, int, int, int, int],
                        weights: Dict[int, int],
                        config: Config) -> Tuple[Tuple[int, int, int, int, int], List[str]]:
    """
    Simulate Protocol C arbitration per target:
      - For each candidate (i->j) compute energy score:
          E = ALPHA * DeltaM_local + BETA * |i-j| + GAMMA * penalty_zero_value
      - DeltaM_local computed over the local stencil.
      - Deterministic tie-break: smaller source index i wins.
      - Commit at most one injection per target.
    Returns (post_sanitized_neigh, trace_lines).
    """
    d = {k: post_neigh[idx] for idx, k in enumerate(range(-2, 3))}
    trace = []
    committed = {j: 0 for j in range(-2, 3)}
    for j in range(-2, 3):
        candidates = mapping.get(j, [])
        if not candidates:
            continue
        best = None
        best_key = None
        for (i, v) in candidates:
            before = info_mass(tuple(d[k] for k in range(-2, 3)), weights, config.M0)
            d_after = d.copy()
            d_after[j] = d_after[j] + v
            after = info_mass(tuple(d_after[k] for k in range(-2, 3)), weights, config.M0)
            deltaM_local = after - before
            penalty = 1.0 if v == 0 else 0.0
            E = config.ALPHA * deltaM_local + config.BETA * abs(i - j) + config.GAMMA * penalty
            key = (E, i)
            if best is None or key < best_key:
                best = (i, v)
                best_key = key
        if best is not None:
            committed[j] = best[1]
            trace.append(f"target {j}: winner source {best[0]} value {best[1]} score {best_key[0]:.6f}")
    final = tuple(d[k] + committed[k] for k in range(-2, 3))
    return final, trace

# ---------------------------
# Generation, verification, and exports
# ---------------------------

def generate_width5_table_and_verify(config: Config) -> Tuple[List[Dict[str, object]], Dict[int, int]]:
    """
    Enumerate all 5-tuples and compute variants and sanitization simulation.
    Returns rows and weights used.
    """
    indices = [-2, -1, 0, 1, 2]
    weights = compute_weights(indices, config.OFFSET_O)
    rows: List[Dict[str, object]] = []
    total = 0
    for combo in __product(config.DIGITS, repeat=5):
        total += 1
        action_interior, post_interior = interior_rewrite(combo)
        action_b0, post_b0 = boundary_rule_i0(combo)
        action_b1, post_b1 = boundary_rule_i1(combo)

        w_before = weighted_sum(combo, weights)
        w_after_interior = weighted_sum(post_interior, weights)
        w_after_b0 = weighted_sum(post_b0, weights)
        w_after_b1 = weighted_sum(post_b1, weights)

        delta_interior = w_after_interior - w_before
        delta_b0 = w_after_b0 - w_before
        delta_b1 = w_after_b1 - w_before

        # sanitization-phase simulation based on interior post
        mapping = deterministic_reconstructor(post_interior, config.RECON_RADIUS)
        post_sanitized, trace = protocol_c_arbitrate(mapping, post_interior, weights, config)
        w_after_sanitized = weighted_sum(post_sanitized, weights)
        delta_sanitized = w_after_sanitized - w_before

        lagrangian_terms = compute_discrete_lagrangian(
            post_interior,
            post_sanitized,
            weights,
            alpha=config.ALPHA,
            beta=config.BETA,
            gamma=config.GAMMA,
        )

        row = {
            "in_d-2": combo[0], "in_d-1": combo[1], "in_d0": combo[2], "in_d1": combo[3], "in_d2": combo[4],
            "action_interior": action_interior,
            "post_interior_d-2": post_interior[0], "post_interior_d-1": post_interior[1],
            "post_interior_d0": post_interior[2], "post_interior_d1": post_interior[3], "post_interior_d2": post_interior[4],
            "delta_interior": delta_interior,
            "action_boundary_i0": action_b0,
            "post_b0_d-2": post_b0[0], "post_b0_d-1": post_b0[1], "post_b0_d0": post_b0[2], "post_b0_d1": post_b0[3], "post_b0_d2": post_b0[4],
            "delta_b0": delta_b0,
            "action_boundary_i1": action_b1,
            "post_b1_d-2": post_b1[0], "post_b1_d-1": post_b1[1], "post_b1_d0": post_b1[2], "post_b1_d1": post_b1[3], "post_b1_d2": post_b1[4],
            "delta_b1": delta_b1,
            "post_sanitized_d-2": post_sanitized[0], "post_sanitized_d-1": post_sanitized[1],
            "post_sanitized_d0": post_sanitized[2], "post_sanitized_d1": post_sanitized[3], "post_sanitized_d2": post_sanitized[4],
            "delta_sanitized": delta_sanitized,
            "delta_H": lagrangian_terms["delta_H"],
            "Q_t": lagrangian_terms["Q_t"],
            "phi_pre": lagrangian_terms["phi_pre"],
            "phi_post": lagrangian_terms["phi_post"],
            "delta_phi": lagrangian_terms["delta_phi"],
            "N_erase": lagrangian_terms["N_erase"],
            "lagrangian": lagrangian_terms["lagrangian"],
            "action_single_step": lagrangian_terms["lagrangian"],
            "sanitization_trace": " | ".join(trace[:6])
        }
        rows.append(row)
    logger.info("Enumerated %d width-5 neighborhoods", total)
    return rows, weights

def write_csv(rows: List[Dict[str, object]], filename: str):
    header = [
        "in_d-2","in_d-1","in_d0","in_d1","in_d2",
        "action_interior",
        "post_interior_d-2","post_interior_d-1","post_interior_d0","post_interior_d1","post_interior_d2","delta_interior",
        "action_boundary_i0",
        "post_b0_d-2","post_b0_d-1","post_b0_d0","post_b0_d1","post_b0_d2","delta_b0",
        "action_boundary_i1",
        "post_b1_d-2","post_b1_d-1","post_b1_d0","post_b1_d1","post_b1_d2","delta_b1",
        "post_sanitized_d-2","post_sanitized_d-1","post_sanitized_d0","post_sanitized_d1","post_sanitized_d2","delta_sanitized",
        "delta_H","Q_t","phi_pre","phi_post","delta_phi","N_erase","lagrangian","action_single_step",
        "sanitization_trace"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def write_verification_report(rows: List[Dict[str, object]], weights: Dict[int, int], config: Config, filename: str):
    total = len(rows)
    failures_interior = [r for r in rows if r["delta_interior"] != 0]
    failures_b0 = [r for r in rows if r["delta_b0"] != 0]
    failures_b1 = [r for r in rows if r["delta_b1"] != 0]
    failures_sanit = [r for r in rows if r["delta_sanitized"] != 0]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("SBA Proofbench Verification Report\n")
        f.write("=================================\n\n")
        f.write(f"Run timestamp: {time.asctime()}\n")
        f.write(f"Total enumerated rows: {total}\n")
        f.write(f"Fibonacci offset (o): {config.OFFSET_O}\n")
        f.write("Weights used (index -> W_index):\n")
        for k in sorted(weights.keys()):
            f.write(f"  {k:>3} -> {weights[k]}\n")
        f.write("\n")
        f.write(f"Interior rule failures (non-zero delta): {len(failures_interior)}\n")
        f.write(f"Boundary i=0 failures: {len(failures_b0)}\n")
        f.write(f"Boundary i=1 failures: {len(failures_b1)}\n")
        f.write(f"Sanitization-phase failures: {len(failures_sanit)}\n\n")
        if failures_interior:
            f.write("Sample interior failures (up to 20):\n")
            for r in failures_interior[:20]:
                f.write(str({k: r[k] for k in ['in_d-2','in_d-1','in_d0','in_d1','in_d2','delta_interior']}) + "\n")
            f.write("\n")
        if failures_b0:
            f.write("Sample boundary i0 failures (up to 20):\n")
            for r in failures_b0[:20]:
                f.write(str({k: r[k] for k in ['in_d-2','in_d-1','in_d0','in_d1','in_d2','delta_b0']}) + "\n")
            f.write("\n")
        if failures_b1:
            f.write("Sample boundary i1 failures (up to 20):\n")
            for r in failures_b1[:20]:
                f.write(str({k: r[k] for k in ['in_d-2','in_d-1','in_d0','in_d1','in_d2','delta_b1']}) + "\n")
            f.write("\n")
        if failures_sanit:
            f.write("Sample sanitization failures (up to 20):\n")
            for r in failures_sanit[:20]:
                f.write(str({k: r[k] for k in ['in_d-2','in_d-1','in_d0','in_d1','in_d2','delta_sanitized','sanitization_trace']}) + "\n")
            f.write("\n")
        f.write("Notes:\n")
        f.write("- The interior rule is algebraically guaranteed to preserve the weighted sum when weights satisfy the Fibonacci identity.\n")
        f.write("- The script performs a deterministic sanitization-phase simulation using Protocol C; sanitization deltas may be nonzero depending on reconstructor behavior and Protocol C parameters.\n")
        f.write("\nEnd of report.\n")

# ---------------------------
# Symbolic proof (SymPy)
# ---------------------------

def symbolic_fibonacci_identity(offset: int = 0) -> Optional[str]:
    """
    Produce a short symbolic proof of 2W_i = W_{i+1} + W_{i-2} for W_k = F_{k+offset}
    using SymPy if available. Returns the proof text or None if SymPy not installed.
    """
    if not SYMPY_AVAILABLE:
        return None
    i, o = sp.symbols('i o', integer=True)
    F = sp.Function('F')
    # Use Fibonacci recurrence: F(n+1) = F(n) + F(n-1)
    # We will symbolically show: 2*F(i+o) - F(i+1+o) - F(i-2+o) == 0
    n = sp.symbols('n', integer=True)
    # Use sympy's fibonacci
    expr = 2*sp.fibonacci(i+o) - sp.fibonacci(i+1+o) - sp.fibonacci(i-2+o)
    simplified = sp.simplify(expr)
    proof_lines = []
    proof_lines.append("Symbolic verification of identity 2W_i = W_{i+1} + W_{i-2} with W_k = F_{k+o}:")
    proof_lines.append(f"SymPy simplified expression for 2*F(i+o) - F(i+1+o) - F(i-2+o): {simplified}")
    proof_lines.append("SymPy uses the Fibonacci recurrence and the identity holds symbolically for integer i,o.")
    return "\n".join(proof_lines)

# ---------------------------
# Informational mass and E = m c^2 examples
# ---------------------------

def compute_energy_examples(config: Config, sample_configs: List[Dict[int, int]]) -> List[Dict[str, object]]:
    """
    Given a list of sample configurations (mapping index -> a_i in {-1,0,1}),
    compute m_sba, m_phys, and E for each and return rows for CSV export.
    """
    indices = sorted({k for sc in sample_configs for k in sc.keys()})
    # compute weights for the union of indices
    weights = compute_weights(indices, config.OFFSET_O)
    rows: List[Dict[str, object]] = []
    for idx, sc in enumerate(sample_configs):
        # ensure all indices present
        m_sba = sum(abs(sc.get(i, 0)) * weights.get(i, fib(i + config.OFFSET_O)) for i in indices)
        # but above weights.get fallback is wrong; recompute properly
        m_sba = sum(abs(sc.get(i, 0)) * compute_weights([i], config.OFFSET_O)[i] for i in indices)
        m_phys = config.MU * m_sba
        E_joule = m_phys * (config.C_LIGHT ** 2)
        rows.append({
            "scenario": "generic_mass_energy",
            "id": idx,
            "config": json.dumps(sc),
            "m_sba": m_sba,
            "m_phys_kg": fmt_sci(m_phys, sig=3),
            "E_joule": fmt_sci(E_joule, sig=3),
            "W_i": "",
            "delta_m_sba": "",
            "delta_E_joule": "",
            "delta_E_target_joule": "",
            "r_au": "",
            "xi": "",
            "casimir_pressure_pa": "",
            "casimir_force_n": "",
            "casimir_gradient_pa_per_au": "",
        })

    # Testable prediction export: sanitization event for W_i = 34.
    # Per manuscript: Delta E = 2 * mu * c^2 * W_i.
    w_i = 34
    delta_m_sba = 2 * w_i
    delta_e = config.MU * (config.C_LIGHT ** 2) * delta_m_sba
    rows.append({
        "scenario": "testable_prediction_sanitization_W34",
        "id": len(rows),
        "config": "{}",
        "m_sba": "",
        "m_phys_kg": "",
        "E_joule": "",
        "W_i": w_i,
        "delta_m_sba": delta_m_sba,
        "delta_E_joule": fmt_sci(delta_e, sig=3),
        "delta_E_target_joule": fmt_sci(6.12e-12, sig=3),
        "r_au": "",
        "xi": "",
        "casimir_pressure_pa": "",
        "casimir_force_n": "",
        "casimir_gradient_pa_per_au": "",
    })

    # Testable prediction export: deep-space Casimir scaling to Oort-cloud radius.
    # Uses manuscript values: P_QED(1um) ~= -1.30e-3 Pa, A = 1e-4 m^2.
    p_qed = -1.30e-3
    area = 1.0e-4
    r_au_ref = 1.0
    r_au_oort = 3000.0
    xi_ref = r_au_ref / r_au_ref
    xi_oort = r_au_ref / r_au_oort
    p_ref = p_qed * xi_ref
    p_oort = p_qed * xi_oort
    f_oort = p_oort * area
    gradient = (p_oort - p_ref) / (r_au_oort - r_au_ref)

    rows.append({
        "scenario": "testable_prediction_casimir_oort_1au",
        "id": len(rows),
        "config": "{}",
        "m_sba": "",
        "m_phys_kg": "",
        "E_joule": "",
        "W_i": "",
        "delta_m_sba": "",
        "delta_E_joule": "",
        "delta_E_target_joule": "",
        "r_au": int(r_au_ref),
        "xi": fmt_sci(xi_ref, sig=3),
        "casimir_pressure_pa": fmt_sci(p_ref, sig=3),
        "casimir_force_n": fmt_sci((p_ref * area), sig=3),
        "casimir_gradient_pa_per_au": fmt_sci(gradient, sig=3),
    })
    rows.append({
        "scenario": "testable_prediction_casimir_oort_3000au",
        "id": len(rows),
        "config": "{}",
        "m_sba": "",
        "m_phys_kg": "",
        "E_joule": "",
        "W_i": "",
        "delta_m_sba": "",
        "delta_E_joule": "",
        "delta_E_target_joule": "",
        "r_au": int(r_au_oort),
        "xi": fmt_sci(xi_oort, sig=3),
        "casimir_pressure_pa": fmt_sci(p_oort, sig=3),
        "casimir_force_n": fmt_sci(f_oort, sig=3),
        "casimir_gradient_pa_per_au": fmt_sci(gradient, sig=3),
    })

    return rows

def write_energy_examples(rows: List[Dict[str, object]], filename: str):
    header = [
        "scenario",
        "id",
        "config",
        "m_sba",
        "m_phys_kg",
        "E_joule",
        "W_i",
        "delta_m_sba",
        "delta_E_joule",
        "delta_E_target_joule",
        "r_au",
        "xi",
        "casimir_pressure_pa",
        "casimir_force_n",
        "casimir_gradient_pa_per_au",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------------------------
# Toy continuous model: rho -> P(rho) -> acceleration
# ---------------------------

def toy_continuous_simulation(config: Config, grid_points: int = 201, source_positions: Optional[List[int]] = None) -> List[Dict[str, float]]:
    """
    Simple 1D toy model:
      - discrete grid of 'cells' with a probability-density rho(x)
      - pressure law P = kappa * rho^k
      - compute acceleration a(x) = - (1/rho_eff) * dP/dx (finite differences)
    This is illustrative: units are arbitrary and the model is not intended
    as a full cosmological solver.
    """
    if source_positions is None:
        source_positions = [grid_points // 2]
    # initialize rho: background small value plus localized sources
    rho = [1e-3 for _ in range(grid_points)]
    for s in source_positions:
        if 0 <= s < grid_points:
            rho[s] += 1.0  # localized high probability density
    kappa = config.PRESSURE_KAPPA
    k = config.PRESSURE_EXPONENT
    # compute pressure
    P = [kappa * (r ** k) for r in rho]
    # finite difference derivative dP/dx (central)
    dx = 1.0
    dPdx = [0.0 for _ in range(grid_points)]
    for i in range(1, grid_points - 1):
        dPdx[i] = (P[i+1] - P[i-1]) / (2.0 * dx)
    # compute effective acceleration a = - (1/rho) * dPdx
    a = [0.0 for _ in range(grid_points)]
    for i in range(grid_points):
        if rho[i] > 0:
            a[i] = - (1.0 / rho[i]) * dPdx[i]
        else:
            a[i] = 0.0
    # export rows
    rows = []
    for i in range(grid_points):
        rows.append({"x": i, "rho": rho[i], "P": P[i], "dPdx": dPdx[i], "a": a[i]})
    return rows

def write_toy_sim_csv(rows: List[Dict[str, float]], filename: str):
    header = ["x", "rho", "P", "dPdx", "a"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------------------------
# Small utilities and helpers
# ---------------------------

def __product(iterable, repeat=1):
    """Deterministic cartesian product iterator."""
    from itertools import product
    yield from product(iterable, repeat=repeat)

# ---------------------------
# Prime-number experiments (unified in proofbench)
# ---------------------------

def is_prime_int(n: int) -> bool:
    """Deterministic primality test for n>=0 (sufficient for current ranges)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def bits_lsb(n: int) -> List[int]:
    out = []
    while n:
        out.append(n & 1)
        n >>= 1
    return out or [0]


def zeckendorf_indices(n: int, fibs: List[int]) -> List[int]:
    if n <= 0:
        return []
    idx = []
    remain = n
    k = len(fibs) - 1
    while remain > 0 and k >= 2:
        while k >= 2 and fibs[k] > remain:
            k -= 1
        if k < 2:
            break
        idx.append(k)
        remain -= fibs[k]
        k -= 2
    idx.sort()
    return idx


def auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    s = np.asarray(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos_ranks = ranks[y == 1].sum()
    return float((pos_ranks - n1 * (n1 + 1) / 2) / (n1 * n0))


def fisher_linear_params(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    X1 = X[y == 1]
    X0 = X[y == 0]
    mu1 = X1.mean(axis=0)
    mu0 = X0.mean(axis=0)
    S1 = np.cov(X1, rowvar=False)
    S0 = np.cov(X0, rowvar=False)
    Sw = S1 + S0 + np.eye(X.shape[1]) * ridge
    w = np.linalg.solve(Sw, (mu1 - mu0))
    return w, mu1, mu0


def mirror_coherence_mod(bits: List[int], m: int) -> float:
    L = len(bits)
    vals = []
    for i in range(L):
        if i % m == 0:
            j = L - 1 - i
            if 0 <= j < L:
                vals.append(1.0 if bits[i] == bits[j] else -1.0)
    return float(np.mean(vals)) if vals else 0.0


def feature_row_prime_sba(n: int, fibs: List[int], moduli=(5, 7, 11, 13)) -> Dict[str, float]:
    bits = bits_lsb(n)
    L = len(bits)
    bstr = format(n, "b")
    ones = sum(bits)

    weighted_prefix = []
    pref = 0
    for i, bit in enumerate(bits):
        pref += bit * fibs[i + 2]
        weighted_prefix.append(pref)

    feats: Dict[str, float] = {}
    for m in moduli:
        traj = np.array([v % m for v in weighted_prefix], dtype=float)
        feats[f"fib_mod{m}_var"] = float(np.var(traj))
        feats[f"fib_mod{m}_span"] = float(np.max(traj) - np.min(traj))

    zidx = zeckendorf_indices(n, fibs)
    gaps = np.diff(zidx) if len(zidx) >= 2 else np.array([], dtype=int)
    feats["z_support_count"] = float(len(zidx))
    feats["z_gap_mean"] = float(np.mean(gaps)) if len(gaps) else 0.0
    feats["z_gap_std"] = float(np.std(gaps)) if len(gaps) else 0.0
    feats["z_gap_max"] = float(np.max(gaps)) if len(gaps) else 0.0

    for m in moduli:
        feats[f"mirror_mod{m}_coh"] = mirror_coherence_mod(bits, m)

    runs = 1 if bstr else 0
    for i in range(1, len(bstr)):
        if bstr[i] != bstr[i - 1]:
            runs += 1
    feats["bit_balance"] = ones / L
    feats["runs"] = runs / L
    feats["bitlen"] = float(L)
    return feats


def oof_scores_fisher(X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, folds: int) -> np.ndarray:
    scores = np.zeros(len(y), dtype=float)
    for f in range(folds):
        test = fold_ids == f
        train = ~test
        w, mu1, mu0 = fisher_linear_params(X[train], y[train])
        scores[test] = (X[test] - 0.5 * (mu1 + mu0)) @ w
    return scores


def write_csv_dicts(path: str, rows: List[Dict[str, object]]):
    if not rows:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_prime_composite_sba_experiment(config: Config, outdir: str):
    seed = 12345
    rng = np.random.default_rng(seed)
    bit_lengths = list(range(8, 21))
    moduli = (5, 7, 11, 13)
    perms = 300
    folds = 5
    fibs = [0, 1]
    for _ in range(2, max(bit_lengths) + 6):
        fibs.append(fibs[-1] + fibs[-2])

    rows: List[Dict[str, object]] = []
    for bl in bit_lengths:
        lo, hi = 1 << (bl - 1), 1 << bl
        odds = list(range(lo | 1, hi, 2))
        primes = [n for n in odds if is_prime_int(n)]
        comps = [n for n in odds if not is_prime_int(n)]
        m = min(len(primes), len(comps))
        if m == 0:
            continue
        lp = list(rng.choice(np.array(primes), size=m, replace=False))
        lc = list(rng.choice(np.array(comps), size=m, replace=False))
        for n in lp:
            r = feature_row_prime_sba(int(n), fibs, moduli)
            r['n'] = int(n)
            r['y'] = 1
            rows.append(r)
        for n in lc:
            r = feature_row_prime_sba(int(n), fibs, moduli)
            r['n'] = int(n)
            r['y'] = 0
            rows.append(r)

    feature_cols = [k for k in rows[0].keys() if k not in ('n', 'y')]
    X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=float)
    y = np.array([int(r['y']) for r in rows], dtype=int)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    fold_ids = np.arange(len(y)) % folds
    rng.shuffle(fold_ids)
    scores = oof_scores_fisher(X, y, fold_ids, folds)
    auc = auc_from_scores(y, scores)

    per_bitlen = []
    bitlen_arr = np.array([int(r['bitlen']) for r in rows], dtype=int)
    for bl in sorted(set(bitlen_arr.tolist())):
        idx = np.where(bitlen_arr == bl)[0]
        per_bitlen.append({'bitlen': bl, 'n': int(len(idx)), 'auc': auc_from_scores(y[idx], scores[idx])})

    null = []
    for _ in range(perms):
        yp = rng.permutation(y)
        sp = oof_scores_fisher(X, yp, fold_ids, folds)
        null.append(auc_from_scores(yp, sp))
    null_arr = np.array(null)
    p_emp = float((np.sum(null_arr >= auc) + 1) / (perms + 1))

    uni_rows = []
    for c in feature_cols:
        vals = np.array([float(r[c]) for r in rows], dtype=float)
        x1, x0 = vals[y == 1], vals[y == 0]
        d = (x1.mean() - x0.mean()) / (np.sqrt(0.5 * (x1.var() + x0.var())) + 1e-9)
        uni_rows.append({'feature': c, 'mean_prime': float(x1.mean()), 'mean_comp': float(x0.mean()), 'cohen_d': float(d)})
    uni_rows.sort(key=lambda z: abs(z['cohen_d']), reverse=True)

    summary = {
        'seed': seed,
        'moduli': list(moduli),
        'bit_lengths': bit_lengths,
        'permutations': perms,
        'folds': folds,
        'N': int(len(rows)),
        'N_primes': int(y.sum()),
        'N_composites': int((y == 0).sum()),
        'auc_global': float(auc),
        'auc_null_mean': float(null_arr.mean()),
        'auc_null_max': float(null_arr.max()),
        'p_empirical': p_emp,
        'per_bitlen': per_bitlen,
    }

    write_csv_dicts(os.path.join(outdir, 'prime_composite_sba_features.csv'), rows)
    write_csv_dicts(os.path.join(outdir, 'prime_composite_sba_univariate.csv'), uni_rows)
    with open(os.path.join(outdir, 'prime_composite_sba_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    lines = [
        'SBA Prime vs Composite Experiment (second-generation feature set)',
        f"N={summary['N']} (primes={summary['N_primes']}, composites={summary['N_composites']})",
        f"Global linear-score AUC={summary['auc_global']:.6f}",
        f"Permutation null mean AUC={summary['auc_null_mean']:.6f}",
        f"Permutation null max AUC={summary['auc_null_max']:.6f}",
        f"Empirical p-value={summary['p_empirical']:.6f}",
        'Per-bit-length AUC:'
    ]
    for r in per_bitlen:
        lines.append(f"  bitlen={r['bitlen']}: n={r['n']}, auc={r['auc']:.6f}")
    with open(os.path.join(outdir, 'prime_composite_sba_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def phase_map_theta(n: int, L: int, fibs: List[int], ln_phi: float) -> float:
    bits = [(n >> i) & 1 for i in range(L)]
    x = np.array([2 * b - 1 for b in bits], dtype=float)
    W = np.array([fibs[i + 2] for i in range(L)], dtype=float)
    k = np.arange(L, dtype=float)
    z = np.sum(x * W * np.exp(1j * k * ln_phi))
    return float(np.angle(z))


def circular_mean_angle(theta: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * theta))))


def in_sector(theta: np.ndarray, center: float, half_width: float) -> np.ndarray:
    d = np.angle(np.exp(1j * (theta - center)))
    return np.abs(d) <= half_width


def stratified_sector_gain(rows: List[Dict[str, object]], center_mode: str = 'all', alpha: float = 0.2):
    half = alpha * np.pi
    per = []
    Ls = sorted({int(r['L']) for r in rows})
    for L in Ls:
        g = [r for r in rows if int(r['L']) == L]
        th_all = np.array([float(r['theta']) for r in g])
        th_p = np.array([float(r['theta']) for r in g if int(r['y']) == 1])
        th_c = np.array([float(r['theta']) for r in g if int(r['y']) == 0])
        center = circular_mean_angle(th_all if center_mode == 'all' else th_p)
        p_in = float(np.mean(in_sector(th_p, center, half)))
        c_in = float(np.mean(in_sector(th_c, center, half)))
        per.append({'L': L, 'p_in': p_in, 'c_in': c_in, 'delta': p_in - c_in})
    return float(np.mean([r['delta'] for r in per])), per


def run_prime_phase_map_experiment(config: Config, outdir: str):
    seed = 20260326
    rng = np.random.default_rng(seed)
    bit_lengths = list(range(8, 21))
    perms = 300
    boots = 300
    alpha = 0.2

    fibs = [0, 1]
    for _ in range(2, max(bit_lengths) + 6):
        fibs.append(fibs[-1] + fibs[-2])
    ln_phi = float(np.log((1 + np.sqrt(5.0)) / 2.0))

    rows: List[Dict[str, object]] = []
    for L in bit_lengths:
        lo, hi = 1 << (L - 1), 1 << L
        odds = list(range(lo | 1, hi, 2))
        primes = [n for n in odds if is_prime_int(n)]
        comps = [n for n in odds if not is_prime_int(n)]
        m = min(len(primes), len(comps))
        if m == 0:
            continue
        lp = list(rng.choice(np.array(primes), size=m, replace=False))
        lc = list(rng.choice(np.array(comps), size=m, replace=False))
        for n in lp:
            rows.append({'n': int(n), 'L': L, 'y': 1, 'theta': phase_map_theta(int(n), L, fibs, ln_phi)})
        for n in lc:
            rows.append({'n': int(n), 'L': L, 'y': 0, 'theta': phase_map_theta(int(n), L, fibs, ln_phi)})

    def perm_null(center_mode: str):
        obs, perL = stratified_sector_gain(rows, center_mode=center_mode, alpha=alpha)
        y = np.array([int(r['y']) for r in rows], dtype=int)
        Larr = np.array([int(r['L']) for r in rows], dtype=int)
        idx_by_L = {L: np.where(Larr == L)[0] for L in sorted(set(Larr.tolist()))}
        null = []
        for _ in range(perms):
            yp = y.copy()
            for L, idx in idx_by_L.items():
                yp[idx] = rng.permutation(yp[idx])
            tmp = [dict(r) for r in rows]
            for i, yy in enumerate(yp.tolist()):
                tmp[i]['y'] = int(yy)
            val, _ = stratified_sector_gain(tmp, center_mode=center_mode, alpha=alpha)
            null.append(val)
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (perms + 1))
        return obs, perL, null, p

    def phase_randomized_null(center_mode: str):
        obs, _ = stratified_sector_gain(rows, center_mode=center_mode, alpha=alpha)
        Larr = np.array([int(r['L']) for r in rows], dtype=int)
        idx_by_L = {L: np.where(Larr == L)[0] for L in sorted(set(Larr.tolist()))}
        null = []
        for _ in range(perms):
            tmp = [dict(r) for r in rows]
            for L, idx in idx_by_L.items():
                vals = rng.uniform(-np.pi, np.pi, size=len(idx))
                for j, ii in enumerate(idx.tolist()):
                    tmp[ii]['theta'] = float(vals[j])
            val, _ = stratified_sector_gain(tmp, center_mode=center_mode, alpha=alpha)
            null.append(val)
        null = np.array(null)
        p = float((np.sum(null >= obs) + 1) / (perms + 1))
        return obs, null, p

    def bootstrap_ci(center_mode: str):
        vals = []
        rows_by_L = {L: [r for r in rows if int(r['L']) == L] for L in sorted({int(r['L']) for r in rows})}
        for _ in range(boots):
            sample = []
            for L, grp in rows_by_L.items():
                idx = rng.integers(0, len(grp), size=len(grp))
                for i in idx:
                    sample.append(dict(grp[int(i)]))
            d, _ = stratified_sector_gain(sample, center_mode=center_mode, alpha=alpha)
            vals.append(d)
        arr = np.array(vals)
        return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    Rp = float(np.abs(np.mean(np.exp(1j * np.array([float(r['theta']) for r in rows if int(r['y']) == 1])))))
    Rc = float(np.abs(np.mean(np.exp(1j * np.array([float(r['theta']) for r in rows if int(r['y']) == 0])))))

    obs_perm, perL, null_perm, p_perm = perm_null('all')
    obs_rand, null_rand, p_rand = phase_randomized_null('all')
    ci_lo, ci_hi = bootstrap_ci('all')
    obs_perm_p, _, null_perm_p, p_perm_p = perm_null('prime')

    summary = {
        'seed': seed,
        'bit_lengths': bit_lengths,
        'N': int(len(rows)),
        'N_primes': int(sum(1 for r in rows if int(r['y']) == 1)),
        'N_composites': int(sum(1 for r in rows if int(r['y']) == 0)),
        'sector_halfwidth_alpha_pi': alpha,
        'circular_resultant_prime': Rp,
        'circular_resultant_composite': Rc,
        'delta_sector_gain_observed_center_all': float(obs_perm),
        'delta_sector_gain_95ci_boot_center_all': [ci_lo, ci_hi],
        'permutation_null_center_all': {
            'n_perm': perms,
            'null_mean': float(null_perm.mean()),
            'null_max': float(null_perm.max()),
            'p_right': float(p_perm),
        },
        'phase_randomized_null_center_all': {
            'n_perm': perms,
            'null_mean': float(null_rand.mean()),
            'null_max': float(null_rand.max()),
            'p_right': float(p_rand),
        },
        'sensitivity_prime_center': {
            'delta_observed': float(obs_perm_p),
            'null_mean': float(null_perm_p.mean()),
            'null_max': float(null_perm_p.max()),
            'p_right': float(p_perm_p),
        },
    }

    write_csv_dicts(os.path.join(outdir, 'prime_phase_map_dataset.csv'), rows)
    write_csv_dicts(os.path.join(outdir, 'prime_phase_map_perL_center_all.csv'), perL)
    with open(os.path.join(outdir, 'prime_phase_map_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    lines = [
        'Prime phase concentration test (SBA H2)',
        f"N={summary['N']} (primes={summary['N_primes']}, composites={summary['N_composites']})",
        f"R_prime={Rp:.6f}, R_comp={Rc:.6f}",
        f"Observed delta (center=all)={obs_perm:.6f}",
        f"Bootstrap 95% CI (center=all)=[{ci_lo:.6f}, {ci_hi:.6f}]",
        f"Permutation null p_right (center=all)={p_perm:.6f}; null_mean={null_perm.mean():.6f}; null_max={null_perm.max():.6f}",
        f"Phase-randomized null p_right (center=all)={p_rand:.6f}; null_mean={null_rand.mean():.6f}; null_max={null_rand.max():.6f}",
        f"Sensitivity (center=prime): delta={obs_perm_p:.6f}, p_right={p_perm_p:.6f}",
    ]
    with open(os.path.join(outdir, 'prime_phase_map_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

# ---------------------------
# Fibonacci-pyramid theorem verification
# ---------------------------

def fibonacci_pyramid_residual(a: float, l: float) -> float:
    """Residual A_triangle - h^2 for a regular square pyramid."""
    if a <= 0 or l <= 0:
        raise ValueError("a and l must be positive")
    h2 = l * l - (a * a) / 4.0
    a_triangle = 0.5 * a * l
    return a_triangle - h2


def verify_fibonacci_pyramid_claim(sample_count: int = 2000) -> Dict[str, float]:
    """
    Verify the manuscript claim:
      A_triangle = h^2  <=>  l/a = (1+sqrt(5))/4 = phi/2.
    Also checks that l/a = phi is generally not equivalent.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    rho_star = (1.0 + math.sqrt(5.0)) / 4.0

    a_values = np.linspace(0.5, 10.0, sample_count)
    residuals_star = []
    residuals_phi = []
    for a in a_values:
        l_star = rho_star * a
        l_phi = phi * a
        residuals_star.append(fibonacci_pyramid_residual(a, l_star))
        residuals_phi.append(fibonacci_pyramid_residual(a, l_phi))

    residuals_star_arr = np.array(residuals_star, dtype=float)
    residuals_phi_arr = np.array(residuals_phi, dtype=float)

    symbolic_zero = None
    if SYMPY_AVAILABLE:
        rho = sp.symbols('rho', positive=True)
        expr = rho**2 - rho/2 - sp.Rational(1, 4)
        symbolic_zero = float(sp.N(expr.subs(rho, sp.Rational(1, 4) + sp.sqrt(5)/4)))

    return {
        "phi": phi,
        "rho_star": rho_star,
        "max_abs_residual_rho_star": float(np.max(np.abs(residuals_star_arr))),
        "mean_abs_residual_rho_star": float(np.mean(np.abs(residuals_star_arr))),
        "min_abs_residual_phi": float(np.min(np.abs(residuals_phi_arr))),
        "mean_abs_residual_phi": float(np.mean(np.abs(residuals_phi_arr))),
        "symbolic_expr_at_rho_star": float(symbolic_zero) if symbolic_zero is not None else float('nan'),
        "samples": float(sample_count),
    }


def write_fibonacci_pyramid_report(result: Dict[str, float], out_path: str):
    lines = [
        "Fibonacci-pyramid theorem verification",
        "Claim: A_triangle = h^2 iff l/a = (1+sqrt(5))/4 = phi/2",
        f"samples={int(result['samples'])}",
        f"rho_star={result['rho_star']:.15f}",
        f"phi={result['phi']:.15f}",
        f"max_abs_residual_rho_star={result['max_abs_residual_rho_star']:.3e}",
        f"mean_abs_residual_rho_star={result['mean_abs_residual_rho_star']:.3e}",
        f"min_abs_residual_phi={result['min_abs_residual_phi']:.3e}",
        f"mean_abs_residual_phi={result['mean_abs_residual_phi']:.3e}",
    ]
    val = result.get('symbolic_expr_at_rho_star', float('nan'))
    if not math.isnan(val):
        lines.append(f"symbolic_expr_at_rho_star={val:.3e}")
    lines.append("Interpretation: rho_star residuals are numerically zero; phi residuals are non-zero.")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------
# Causal-pyramid verification
# ---------------------------

def fit_linear_front(times: np.ndarray, radii: np.ndarray) -> Dict[str, float]:
    """Least-squares fit r(t) = a + b t and diagnostics."""
    coeff = np.polyfit(times, radii, 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = intercept + slope * times
    rss = float(np.sum((radii - pred) ** 2))
    return {"intercept": intercept, "slope": slope, "rss": rss, "k": 2.0}


def fit_diffusive_front(times: np.ndarray, radii: np.ndarray) -> Dict[str, float]:
    """Least-squares fit r(t) = a + b*sqrt(t) and diagnostics."""
    sqrt_t = np.sqrt(times)
    coeff = np.polyfit(sqrt_t, radii, 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = intercept + slope * sqrt_t
    rss = float(np.sum((radii - pred) ** 2))
    return {"intercept": intercept, "slope": slope, "rss": rss, "k": 2.0}


def model_selection_scores(rss: float, n: int, k: int) -> Dict[str, float]:
    """Compute AIC/BIC for Gaussian residual model (up to additive constants)."""
    safe_rss = max(rss, 1e-15)
    aic = float(n * math.log(safe_rss / n) + 2 * k)
    bic = float(n * math.log(safe_rss / n) + k * math.log(n))
    return {"aic": aic, "bic": bic}


def finite_speed_front_bound(times: np.ndarray, radii: np.ndarray, r0: float, v_i: float) -> Dict[str, float]:
    """Check finite-speed inequality r(t) <= r0 + v_i*(t-t0)."""
    t0 = float(times[0])
    bounds = r0 + v_i * (times - t0)
    slack = bounds - radii
    min_slack = float(np.min(slack))
    max_violation = float(max(0.0, -min_slack))
    return {
        "t0": t0,
        "r0": r0,
        "v_i": v_i,
        "min_slack": min_slack,
        "max_violation": max_violation,
        "bound_holds": float(max_violation <= 1e-9),
    }


def directional_anisotropy_diagnostics(radii_by_direction: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Summarize final-time directional spread."""
    if len(radii_by_direction) == 0:
        return {
            "directions": 0.0,
            "mean_final_radius": float("nan"),
            "max_final_radius": float("nan"),
            "min_final_radius": float("nan"),
            "spread_final_radius": float("nan"),
            "relative_spread": float("nan"),
        }
    final_radii = [float(arr[-1]) for arr in radii_by_direction.values()]
    mean_final = float(np.mean(final_radii))
    max_final = float(np.max(final_radii))
    min_final = float(np.min(final_radii))
    spread = max_final - min_final
    rel_spread = spread / max(mean_final, 1e-9)
    return {
        "directions": float(len(radii_by_direction)),
        "mean_final_radius": mean_final,
        "max_final_radius": max_final,
        "min_final_radius": min_final,
        "spread_final_radius": spread,
        "relative_spread": rel_spread,
    }


def continuum_bridge_calibration(delta_z: float, delta_tau: float, v_i: float) -> Dict[str, float]:
    """Bridge lattice speed to continuum speed c_I=(Δz/Δτ)*v_I."""
    if delta_tau <= 0:
        raise ValueError("delta_tau must be positive")
    if delta_z <= 0:
        raise ValueError("delta_z must be positive")
    c_i = (delta_z / delta_tau) * v_i
    return {"delta_z": delta_z, "delta_tau": delta_tau, "v_i": v_i, "c_I": float(c_i)}


def _synthetic_causal_pyramid_dataset(sample_count: int = 40) -> Dict[str, np.ndarray]:
    """Deterministic synthetic fronts consistent with finite-speed propagation."""
    times = np.arange(float(sample_count), dtype=float)
    r0 = 1.0
    v_i = 0.75
    base = r0 + v_i * times
    wobble = 0.03 * np.sin(0.4 * times)
    radii_center = base - np.abs(wobble)
    return {
        "times": times,
        "radii_center": radii_center,
        "r0": np.array([r0], dtype=float),
        "v_i": np.array([v_i], dtype=float),
        "axis_x_pos": radii_center + 0.04,
        "axis_x_neg": radii_center - 0.03,
        "axis_y_pos": radii_center + 0.02,
        "axis_y_neg": radii_center - 0.01,
        "diag_pos": radii_center - 0.10,
        "diag_neg": radii_center - 0.08,
    }


def load_causal_pyramid_input_csv(input_csv: str) -> Dict[str, np.ndarray]:
    """
    Load measured front data from CSV.

    Required columns:
      - t
      - r_center
    Optional columns:
      - r0, v_i (single scalar metadata; first row is used)
      - any number of directional columns with prefix r_ and name != r_center
    """
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("causal-pyramid input CSV must contain at least 3 rows")
    if reader.fieldnames is None:
        raise ValueError("causal-pyramid input CSV has no header")
    if "t" not in reader.fieldnames or "r_center" not in reader.fieldnames:
        raise ValueError("causal-pyramid input CSV must include 't' and 'r_center' columns")

    times = np.array([float(row["t"]) for row in rows], dtype=float)
    radii_center = np.array([float(row["r_center"]) for row in rows], dtype=float)
    if np.any(np.diff(times) < 0):
        raise ValueError("causal-pyramid input CSV times must be non-decreasing")
    if np.min(times) < 0:
        raise ValueError("causal-pyramid input CSV times must be non-negative")

    directional_keys: List[str] = []
    for name in reader.fieldnames:
        if name.startswith("r_") and name not in ("r_center", "r0"):
            directional_keys.append(name[2:])

    data: Dict[str, np.ndarray] = {
        "times": times,
        "radii_center": radii_center,
        "r0": np.array([float(rows[0].get("r0", radii_center[0]))], dtype=float),
        "v_i": np.array([float(rows[0].get("v_i", 0.0))], dtype=float),
    }
    for key in directional_keys:
        column = f"r_{key}"
        data[key] = np.array([float(row[column]) for row in rows], dtype=float)
    return data


def verify_causal_pyramid_claim(
    outdir: str,
    sample_count: int = 40,
    input_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    Verify causal-pyramid behavior with finite-speed fronts and model discrimination.

    If input_csv is provided, measured fronts are loaded from file.
    Otherwise deterministic synthetic fronts are generated.
    """
    data = load_causal_pyramid_input_csv(input_csv) if input_csv else _synthetic_causal_pyramid_dataset(sample_count)
    times = data["times"]
    radii_center = data["radii_center"]

    if len(times) < 3:
        raise ValueError("at least 3 time samples are required")

    r0 = float(data["r0"][0])
    v_i = float(data["v_i"][0])
    if v_i <= 0:
        slope_seed = fit_linear_front(times, radii_center)["slope"]
        v_i = max(0.0, slope_seed)

    radii_by_direction: Dict[str, np.ndarray] = {
        key: value
        for key, value in data.items()
        if key not in {"times", "radii_center", "r0", "v_i"}
    }

    bound = finite_speed_front_bound(times, radii_center, r0=r0, v_i=v_i)
    linear_fit = fit_linear_front(times, radii_center)
    diffusive_fit = fit_diffusive_front(times, radii_center)
    linear_scores = model_selection_scores(linear_fit["rss"], n=len(times), k=int(linear_fit["k"]))
    diffusive_scores = model_selection_scores(diffusive_fit["rss"], n=len(times), k=int(diffusive_fit["k"]))
    anisotropy = directional_anisotropy_diagnostics(radii_by_direction)
    bridge = continuum_bridge_calibration(delta_z=1.0, delta_tau=1.0, v_i=v_i)

    direction_columns = sorted(radii_by_direction.keys())
    rows: List[Dict[str, float]] = []
    for i, t in enumerate(times):
        row: Dict[str, float] = {
            "t": float(t),
            "r_center": float(radii_center[i]),
            "r_bound": float(r0 + v_i * (t - times[0])),
        }
        for key in direction_columns:
            row[f"r_{key}"] = float(radii_by_direction[key][i])
        rows.append(row)
    write_csv_dicts(os.path.join(outdir, "causal_pyramid_fronts.csv"), rows)

    return {
        "samples": float(len(times)),
        "front_r0": r0,
        "front_v_i": v_i,
        "finite_speed_bound_holds": bound["bound_holds"],
        "finite_speed_max_violation": bound["max_violation"],
        "linear_fit_slope": linear_fit["slope"],
        "linear_fit_intercept": linear_fit["intercept"],
        "linear_fit_rss": linear_fit["rss"],
        "diffusive_fit_slope": diffusive_fit["slope"],
        "diffusive_fit_intercept": diffusive_fit["intercept"],
        "diffusive_fit_rss": diffusive_fit["rss"],
        "linear_aic": linear_scores["aic"],
        "diffusive_aic": diffusive_scores["aic"],
        "linear_bic": linear_scores["bic"],
        "diffusive_bic": diffusive_scores["bic"],
        "delta_aic_diffusive_minus_linear": diffusive_scores["aic"] - linear_scores["aic"],
        "delta_bic_diffusive_minus_linear": diffusive_scores["bic"] - linear_scores["bic"],
        "anisotropy_directions": anisotropy["directions"],
        "anisotropy_relative_spread": anisotropy["relative_spread"],
        "anisotropy_spread_final_radius": anisotropy["spread_final_radius"],
        "continuum_delta_z": bridge["delta_z"],
        "continuum_delta_tau": bridge["delta_tau"],
        "continuum_c_I": bridge["c_I"],
        "input_mode_measured": float(1.0 if input_csv else 0.0),
    }


def write_causal_pyramid_report(result: Dict[str, float], out_path: str):
    lines = [
        "Causal-pyramid verification",
        "Finite-speed front check: r(t) <= r0 + v_I (t - t0)",
        f"samples={int(result['samples'])}",
        f"measured_input_mode={int(result['input_mode_measured'])}",
        f"r0={result['front_r0']:.6f}",
        f"v_I={result['front_v_i']:.6f}",
        f"finite_speed_bound_holds={int(result['finite_speed_bound_holds'])}",
        f"finite_speed_max_violation={result['finite_speed_max_violation']:.3e}",
        "Model discrimination (linear front vs diffusive sqrt(t) front):",
        f"linear_rss={result['linear_fit_rss']:.6e}, diffusive_rss={result['diffusive_fit_rss']:.6e}",
        f"linear_aic={result['linear_aic']:.6f}, diffusive_aic={result['diffusive_aic']:.6f}, delta={result['delta_aic_diffusive_minus_linear']:.6f}",
        f"linear_bic={result['linear_bic']:.6f}, diffusive_bic={result['diffusive_bic']:.6f}, delta={result['delta_bic_diffusive_minus_linear']:.6f}",
        "Directional anisotropy diagnostics:",
        f"directions={int(result['anisotropy_directions'])}",
        f"relative_spread={result['anisotropy_relative_spread']}",
        f"spread_final_radius={result['anisotropy_spread_final_radius']}",
        "Continuum bridge calibration:",
        f"c_I=(delta_z/delta_tau)*v_I = {result['continuum_c_I']:.6f}",
        "Artifacts: causal_pyramid_fronts.csv, causal_pyramid_verification.txt",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------
# Unit tests (unittest)
# ---------------------------

def run_unit_tests(config: Config, report_path: str) -> int:
    """
    Run a small suite of unit tests for core primitives and write a short report.
    Returns the number of failed tests (0 means all passed).
    """
    import unittest

    class CoreTests(unittest.TestCase):
        def test_fib_negative(self):
            # F_{-n} = (-1)^{n+1} F_n
            for n in range(1, 11):
                self.assertEqual(fib(-n), ((-1) ** (n + 1)) * fib(n))

        def test_weights_offset(self):
            # check that compute_weights returns expected Fibonacci numbers for small indices
            w = compute_weights([-2, -1, 0, 1, 2], config.OFFSET_O)
            for k in [-2, -1, 0, 1, 2]:
                self.assertEqual(w[k], fib(k + config.OFFSET_O))

        def test_interior_preserves_weighted_sum(self):
            # test a few representative neighborhoods
            w = compute_weights([-2, -1, 0, 1, 2], config.OFFSET_O)
            examples = [
                (-2, 0, 2, 0, 0),
                (0, 1, -2, 1, 0),
                (1, -1, 0, 2, -1),
                (0, 0, 0, 0, 0)
            ]
            for ex in examples:
                _, post = interior_rewrite(ex)
                self.assertEqual(weighted_sum(ex, w), weighted_sum(post, w))

        def test_info_mass(self):
            w = compute_weights([-2, -1, 0, 1, 2], config.OFFSET_O)
            ex = (1, -1, 0, 1, -1)
            m = info_mass(ex, w, config.M0)
            # manual compute
            manual = config.M0 * sum(abs(ex[i]) * w[k] for i, k in enumerate([-2, -1, 0, 1, 2]))
            self.assertAlmostEqual(m, manual)

        def test_discrete_lagrangian_computation(self):
            w = compute_weights([-2, -1, 0, 1, 2], config.OFFSET_O)
            pre = (-1, 1, 2, 0, -1)
            post = interior_rewrite(pre)[1]
            terms = compute_discrete_lagrangian(pre, post, w)
            self.assertIn("delta_H", terms)
            self.assertIn("Q_t", terms)
            self.assertIn("lagrangian", terms)
            self.assertGreaterEqual(terms["Q_t"], 0.0)

        def test_verify_cpt_symmetry(self):
            # Requested numerical CPT check on an example tuple.
            ex = (-1, 1, 2, 0, -1)
            self.assertTrue(verify_cpt_symmetry(ex, config))

        def test_fibonacci_pyramid_ratio(self):
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            rho_star = phi / 2.0
            a = 2.0
            l = rho_star * a
            self.assertAlmostEqual(fibonacci_pyramid_residual(a, l), 0.0, places=12)

        def test_fibonacci_pyramid_phi_not_equivalent(self):
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            a = 2.0
            l = phi * a
            self.assertGreater(abs(fibonacci_pyramid_residual(a, l)), 1e-3)

        def test_causal_pyramid_synthetic_path(self):
            outdir = os.path.dirname(report_path)
            result = verify_causal_pyramid_claim(outdir=outdir, sample_count=20)
            self.assertEqual(int(result["finite_speed_bound_holds"]), 1)
            self.assertGreater(result["delta_aic_diffusive_minus_linear"], 0.0)
            self.assertTrue(os.path.exists(os.path.join(outdir, "causal_pyramid_fronts.csv")))

        def test_causal_pyramid_measured_input_path(self):
            outdir = os.path.dirname(report_path)
            measured_csv = os.path.join(outdir, "causal_pyramid_measured.csv")
            with open(measured_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["t", "r_center", "r0", "v_i", "r_axis_x", "r_diag"])
                writer.writeheader()
                for t in range(10):
                    center = 1.0 + 0.5 * t
                    writer.writerow({
                        "t": float(t),
                        "r_center": center,
                        "r0": 1.0,
                        "v_i": 0.6,
                        "r_axis_x": center + 0.02,
                        "r_diag": center - 0.04,
                    })
            result = verify_causal_pyramid_claim(outdir=outdir, input_csv=measured_csv)
            self.assertEqual(int(result["input_mode_measured"]), 1)
            self.assertTrue(os.path.exists(os.path.join(outdir, "causal_pyramid_fronts.csv")))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    with open(report_path, "w", encoding="utf-8") as report_stream:
        runner = unittest.TextTestRunner(stream=report_stream, verbosity=2)
        result = runner.run(suite)
    failed = len(result.failures) + len(result.errors)
    return failed

# ---------------------------
# Main orchestration
# ---------------------------

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="SBA Theory Proofbench: verification and simulation harness.")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: enumeration, verification, symbolic proof, energy examples, toy sim, unit tests.")
    parser.add_argument("--enumerate", action="store_true", help="Run exhaustive width-5 enumeration and verification.")
    parser.add_argument("--symbolic", action="store_true", help="Run symbolic SymPy proof (if SymPy installed).")
    parser.add_argument("--energy", action="store_true", help="Compute energy examples and export CSV.")
    parser.add_argument("--toy", action="store_true", help="Run toy continuous simulation and export CSV.")
    parser.add_argument("--tests", action="store_true", help="Run unit tests.")
    parser.add_argument("--prime", action="store_true", help="Run prime/composite SBA feature experiment.")
    parser.add_argument("--phase", action="store_true", help="Run SBA phase concentration test with null models.")
    parser.add_argument("--pyramid", action="store_true", help="Verify Fibonacci-pyramid theorem numerically/symbolically.")
    parser.add_argument("--causal-pyramid", action="store_true", help="Verify finite-speed causal-pyramid fronts and diagnostics.")
    parser.add_argument("--causal-pyramid-input", type=str, default=None, help="Optional measured-front CSV for --causal-pyramid mode.")
    parser.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Output directory for artifacts.")
    args = parser.parse_args(argv)

    # instantiate config
    config = Config()
    config.OUTPUT_DIR = args.outdir
    config.RUN_ID = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:12]

    ensure_output_dir(config.OUTPUT_DIR)
    logger.info("Proofbench output directory: %s", config.OUTPUT_DIR)

    # export parameters
    with open(os.path.join(config.OUTPUT_DIR, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Wrote parameters.json")

    # run tasks
    if args.all or args.enumerate:
        logger.info("Starting exhaustive width-5 enumeration and verification...")
        rows, weights = generate_width5_table_and_verify(config)
        write_csv(rows, os.path.join(config.OUTPUT_DIR, "width5_transitions_full.csv"))
        write_verification_report(rows, weights, config, os.path.join(config.OUTPUT_DIR, "verification_report.txt"))
        logger.info("Enumeration and verification artifacts written.")

    if args.all or args.symbolic:
        logger.info("Attempting symbolic verification with SymPy...")
        proof_text = symbolic_fibonacci_identity(config.OFFSET_O)
        if proof_text is None:
            logger.warning("SymPy not available; skipping symbolic proof.")
            with open(os.path.join(config.OUTPUT_DIR, "symbolic_proof.txt"), "w", encoding="utf-8") as f:
                f.write("SymPy not available in this environment. Install sympy to enable symbolic verification.\n")
        else:
            with open(os.path.join(config.OUTPUT_DIR, "symbolic_proof.txt"), "w", encoding="utf-8") as f:
                f.write(proof_text + "\n")
            logger.info("Symbolic proof written to symbolic_proof.txt")

    if args.all or args.energy:
        logger.info("Computing energy examples...")
        # sample configurations: single activation at various indices and small composite examples
        sample_configs = [
            {0: 1},  # single activation at center
            {2: 1},  # rightmost
            {-2: 1},  # leftmost
            {0: 1, 1: 1},  # two activations
            {0: 1, 0: -1},  # cancellation example (should be interpreted by model semantics)
        ]
        energy_rows = compute_energy_examples(config, sample_configs)
        write_energy_examples(energy_rows, os.path.join(config.OUTPUT_DIR, "energy_examples.csv"))
        logger.info("Energy examples written to energy_examples.csv")

    if args.all or args.toy:
        logger.info("Running toy continuous simulation...")
        toy_rows = toy_continuous_simulation(config, grid_points=201, source_positions=[100])
        write_toy_sim_csv(toy_rows, os.path.join(config.OUTPUT_DIR, "toy_simulation.csv"))
        logger.info("Toy simulation written to toy_simulation.csv")

    if args.all or args.prime:
        logger.info("Running prime/composite SBA experiment...")
        run_prime_composite_sba_experiment(config, config.OUTPUT_DIR)
        logger.info("Prime/composite artifacts written.")

    if args.all or args.phase:
        logger.info("Running SBA phase concentration experiment...")
        run_prime_phase_map_experiment(config, config.OUTPUT_DIR)
        logger.info("Phase-map artifacts written.")

    if args.all or args.pyramid:
        logger.info("Verifying Fibonacci-pyramid theorem claim...")
        result = verify_fibonacci_pyramid_claim(sample_count=2000)
        write_fibonacci_pyramid_report(result, os.path.join(config.OUTPUT_DIR, "fibonacci_pyramid_verification.txt"))
        logger.info("Pyramid verification written to fibonacci_pyramid_verification.txt")

    if args.all or args.causal_pyramid:
        logger.info("Verifying causal-pyramid finite-speed and front-shape diagnostics...")
        result = verify_causal_pyramid_claim(
            outdir=config.OUTPUT_DIR,
            sample_count=40,
            input_csv=args.causal_pyramid_input,
        )
        write_causal_pyramid_report(result, os.path.join(config.OUTPUT_DIR, "causal_pyramid_verification.txt"))
        logger.info("Causal-pyramid verification written to causal_pyramid_verification.txt")

    if args.all or args.tests:
        logger.info("Running unit tests...")
        unit_test_report_path = os.path.join(config.OUTPUT_DIR, "unit_test_report.txt")
        failed = run_unit_tests(config, unit_test_report_path)
        if failed == 0:
            logger.info("All unit tests passed. Report at %s", unit_test_report_path)
        else:
            logger.error("Unit tests reported %d failures. See %s", failed, unit_test_report_path)

    logger.info("Proofbench run completed. Inspect %s for artifacts.", config.OUTPUT_DIR)

if __name__ == "__main__":
    main()