# Specular Bit Architecture (SBA) — Repository Guide

This repository contains the manuscript, the mathematical formalization, and the reproducibility pipeline for the **Specular Bit Architecture (SBA)** framework.

### The Core Concept: From Cosmological Theory to AI Hardware
The SBA framework originated from a broader cosmological and physical theory exploring how localized informational states and causal structures emerge from underlying quantum/thermodynamic dynamics. By mapping these continuous geometric principles into a discrete digital framework, I derived the SBA. 

Remarkably, this dual-carrier specular logic closely mirrors the fundamental coding mechanisms of biological DNA (e.g., complementary base pairing). As a result, the SBA serves as a **bridge between theoretical physics, DNA data storage, and the next generation of AI-facing neuromorphic hardware**.

This repository provides the rigorous Python proofbench that executes all the formal validations, structural enumerations, and energy-mapping calculations described in the manuscript.

It is organized to strictly separate:
- **Formal results** (theorems/lemmas/proofs in the PDF/`main.tex`),
- **Computational certificates** (finite exhaustive checks in the Python proofbench),
- **Modeling/diagnostic outputs** (hypothesis-layer experiments and deep-space/biological mappings).

---

## Repository Layout

- `main.pdf` — The compiled IEEE/Nature-format manuscript.
- `main.tex` — Main LaTeX manuscript source.
- `references.bib` — Bibliography used by the manuscript.
- `sba_theory_proofbench.py` — The core Python verification and simulation harness.
- `scripts/sba_theory_proofbench.py` — Full proofbench implementation (if using directory structure).
- `outputs/` — Generated CSV/TXT/JSON artifacts from the proofbench.
- `prism-uploads/`, `uploads/` — Working assets and imported materials.

---

## What the Proofbench Validates and Generates

The Python script (`sba_theory_proofbench.py`) performs all the heavy lifting to prove the claims made in the paper. It generates all manuscript-linked computational artifacts.

### Core finite validation pipeline
Triggered by `--enumerate` (or `--all`):
1. **Exhaustive width-5 state enumeration:** Computes all `5^5 = 3125` local neighborhood states.
2. **Rewrite Simulation:** Applies interior and boundary exact rewrite rules.
3. **Weighted-Sum Delta Checks:** Proves exact local preservation of informational mass.
4. **Arbitration Simulation:** Runs deterministic sanitization + Protocol C arbitration.
5. **Pattern Exports:** Canonical-pattern and critical-pair matrix exports.
6. **Validation Bundles:** Generates consolidated machine/human validation bundles.

Primary outputs:
- `outputs/width5_transitions_full.csv`
- `outputs/verification_report.txt`
- `outputs/validation_summary.json`
- `outputs/required_results_bundle.json`
- `outputs/required_results_bundle.txt`
- `outputs/canonical_patterns_len5.csv`
- `outputs/critical_pairs_full.csv`
- `outputs/critical_pairs_counts_by_patternA.csv`

### Optional Modules (Physics, AI, and Mathematics)
- `--symbolic`: Symbolic Fibonacci-identity verification via SymPy (`symbolic_proof.txt`).
- `--energy`: Bookkeeping and physical testable-value exports (e.g., Casimir effect scale, E=mc^2 mapping) (`energy_examples.csv`).
- `--toy`: Toy 1D continuous model export bridging discrete/continuous states (`toy_simulation.csv`).
- `--tests`: Suite of unit tests for core primitives (`unit_test_report.txt`).
- `--prime`: Prime/composite SBA feature experiment.
- `--phase`: Phase-map concentration and null-model experiment.
- `--evo-bridge`: Transition-derived bridge quantities (`Q`, class summary, `L`).
- `--causal-pyramid --causal-pyramid-input <csv>`: Measured-front diagnostics.

Additionally, each run writes:
- `outputs/parameters.json` (run/config snapshot for reproducibility)
- `outputs/README_GENERATION.txt` (artifact index + SHA-256 hashes to prevent tampering)

---

## Requirements

- **Python 3.8+**
- Built-in/available packages used by this repo: `numpy`, optional `sympy` (for symbolic proofs)
- No virtual environment is strictly required.

---

## How to Run

Run the script directly from the repository root.

### 1) Full pipeline (All validations and physics mappings)
```bash
python3 sba_theory_proofbench.py --all --outdir outputs
```

### 2) Core validation only (Recommended for Peer-Reviewers)
```bash
python3 sba_theory_proofbench.py --enumerate --outdir outputs
```

### 3) Optional modules (Granular execution)
```bash
python3 sba_theory_proofbench.py --symbolic --energy --toy --tests --outdir outputs
python3 sba_theory_proofbench.py --prime --phase --evo-bridge --outdir outputs
python3 sba_theory_proofbench.py --causal-pyramid --causal-pyramid-input path/to/fronts.csv --outdir outputs
```

### 4) Causal-pyramid with full run
If you want causal-pyramid diagnostics during full execution:
```bash
python3 sba_theory_proofbench.py --all --causal-pyramid-input path/to/fronts.csv --outdir outputs
```

---

## CLI Options

```bash
python3 sba_theory_proofbench.py [OPTIONS]
```

- `--all` : run full pipeline modules
- `--enumerate` : run exhaustive width-5 verification pipeline
- `--symbolic` : symbolic Fibonacci identity check
- `--energy` : informational mass/energy physics examples
- `--toy` : toy continuous simulation
- `--tests` : core primitive unit tests
- `--prime` : prime/composite diagnostics
- `--phase` : phase-map diagnostics
- `--evo-bridge` : bridge quantity estimation
- `--causal-pyramid` : causal-pyramid diagnostics
- `--causal-pyramid-input <csv>` : input CSV (`t`, `r_center` required)
- `--outdir <path>` : output directory (default: `outputs`)

---

## Reproducible Reviewer Workflow

We adhere strictly to reproducible research standards.
1. Generate baseline artifacts on your local machine:
```bash
python3 sba_theory_proofbench.py --enumerate --outdir outputs
```
2. Check `outputs/verification_report.txt` for finite-check counts.
3. Parse `outputs/required_results_bundle.json` for machine-readable validation values.
4. Use `outputs/README_GENERATION.txt` hashes for provenance locking and cross-referencing with our published hashes.
5. (Optional) run `--tests` and compare `outputs/unit_test_report.txt`.

---

## Interpreting Outputs Correctly

- Proofbench outputs are **finite computational certificates** and diagnostics.
- They **support** theorem-level claims but do not replace the formal mathematical proofs present in `main.tex/main.pdf`.
- Extracted constants/certificates are **rule/policy/radius specific** unless re-certified under changed assumptions via the script parameters.
- Modeling and AI-facing modules (e.g., DNA mapping, energy calculations) are hypothesis-layer diagnostics and should be read under the assumptions explicitly declared in the manuscript.

---

## Troubleshooting

- If the *symbolic proof* is skipped, it means `sympy` is unavailable in your environment; the script will write an explanatory `symbolic_proof.txt` and continue safely.
- If *causal-pyramid* fails, ensure the input CSV exists and contains the required columns: `t`, `r_center`.
- To verify execution integrity, re-run with the same command and compare `README_GENERATION.txt` hashes.