# Specular Bit Architecture (SBA) — SBA_DNA Proofbench

**Author:** Marco Oppido  
**Contact:** [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com)

## Overview
This repository contains the SBA_DNA theory (Specular Bit Architecture) and a single, authoritative verification tool: the proofbench script `scripts/sba_theory_proofbench.py`. The script is the only program required to reproduce every check, validation, and modeling artifact referenced in the manuscript. It performs exhaustive local enumerations, symbolic identity checks (when SymPy is available), sanitization and Protocol C arbitration simulations, informational mass–energy bookkeeping examples, toy continuous-model exports, and unit tests.

This README explains how the theory is testable through that script, how to run it, what outputs to expect, and how to interpret the results for reproducible review.

## Repository layout

```text
SBA_DNA.pdf                             # Manuscript (cosmology + SBA formalism)
scripts/
  sba_theory_proofbench.py              # Single-file proofbench (entrypoint)
figures/                                # Figures referenced by the manuscript
outputs/                                # Generated artifacts (CSV/TXT/JSON) — created by the proofbench
README.md                               # This file
LICENSE                                 # Add your chosen license before public release
```

**Important:** the repository intentionally exposes one canonical script (`scripts/sba_theory_proofbench.py`). That script generates all verification artifacts; no additional scripts are required.

## Purpose of the proofbench
The proofbench is designed to make the SBA formal and modeling claims directly testable and reproducible. It implements the same primitives and semantics used in the manuscript:
* Fibonacci positional weights $W_i = F_{i+o}$ (supports negative indices).
* Interior and boundary rewrite rules that preserve signed weighted sums.
* Sanitization operator mapping transient ambiguous microstates to canonical states.
* Deterministic reconstruction and Protocol C arbitration (local, deterministic scoring and tie-breaking).
* Lexicographic measure and bounded-propagation checks.
* Informational mass bookkeeping and example $E=mc^2$ mappings.
* Optional symbolic verification of algebraic identities (requires SymPy).

The script centralizes parameter management and writes a `parameters.json` file for reproducibility.

## Quickstart — exact commands

### Prerequisites
* Python 3.8+
* Optional but recommended: `sympy` for symbolic checks
* Typical scientific packages: `numpy` (the script runs with standard library + NumPy; SymPy is optional)

### Run the full pipeline (from repository root)
```bash
# Optional: install SymPy for symbolic verification
python3 -m pip install --user sympy

# Create outputs directory (if not present)
mkdir -p outputs

# Run the complete proofbench and write artifacts to outputs/
python3 scripts/sba_theory_proofbench.py --all --outdir outputs
```

### Run selected modules
```bash
# Exhaustive width-5 enumeration and verification only
python3 scripts/sba_theory_proofbench.py --enumerate --outdir outputs

# Symbolic Fibonacci identity check only (requires SymPy)
python3 scripts/sba_theory_proofbench.py --symbolic --outdir outputs

# Mass/energy examples only
python3 scripts/sba_theory_proofbench.py --energy --outdir outputs

# Unit tests only
python3 scripts/sba_theory_proofbench.py --tests --outdir outputs
```

### Main CLI options
* `--all` : run the complete pipeline
* `--enumerate` : exhaustive width‑5 enumeration + verification reports
* `--symbolic` : symbolic Fibonacci identity check (requires SymPy)
* `--energy` : informational mass/energy examples
* `--toy` : toy continuous simulation export
* `--tests` : unit test suite
* `--prime` / `--phase` : exploratory experiments included in the script
* `--outdir <path>` : output directory (default: `outputs`)

## Expected outputs and how to read them
When you run `--all` the script writes deterministic artifacts to the chosen `outdir`. Representative files and their purpose:

### Formal / verification artifacts
* `width5_transitions_full.csv` — exhaustive local transition table (all 5‑tuples and their interior/boundary/sanitized outcomes). Use this to verify local rewrite classes and to reproduce the exhaustive enumeration counts cited in the manuscript.
* `verification_report.txt` — human‑readable summary of algebraic checks (interior rule preservation, boundary checks) and sample failing rows if any.
* `symbolic_proof.txt` — symbolic verification of the Fibonacci identity (present only if SymPy is installed).
* `unit_test_report.txt` — unit-test results for core primitives.
* `parameters.json` — exact parameter set used for the run (use this plus the repository commit hash to reproduce results).

### Modeling / simulation artifacts
* `energy_examples.csv` — informational mass → physical proxy → $E=mc^2$ examples (illustrative; requires independent calibration of `MU` for physical claims).
* `toy_simulation.csv` — toy continuous-model outputs (illustrative pressure/propagation examples).

### Exploratory experiment artifacts (if run)
* `prime_composite_sba_*` and `prime_phase_map_*` files — exploratory statistical/modeling outputs; treat as hypothesis-level artifacts.

## How to interpret verification results
* **Interior rule preservation:** the interior rewrite is algebraically derived from the Fibonacci identity; the verification report should show zero interior-rule failures when the canonical offset `OFFSET_O` is set as in the manuscript (default 2). If failures appear, inspect `verification_report.txt` for sample rows and confirm the `parameters.json` used.
* **Sanitization deltas:** sanitization and reconstruction are simulated with a deterministic reconstructor and Protocol C; nonzero deltas in sanitized outcomes reflect reconstructor behavior and Protocol C parameters (these are modeling choices, not algebraic contradictions).
* **Symbolic check:** if SymPy is available, the script produces a symbolic simplification confirming the identity $2W_i = W_{i+1} + W_{i-2}$ for $W_k = F_{k+o}$.

## Reproducibility and provenance
* **Deterministic runs:** the proofbench is deterministic given the same code and `parameters.json`. Each run writes `parameters.json` and timestamps into the output directory.
* **Exact reproduction:** to reproduce a published run, record:
  1. repository commit hash (or release tag),
  2. the `parameters.json` produced by the run, and
  3. the exact command-line used.
* **Parameter editing:** change the `Config` dataclass at the top of `scripts/sba_theory_proofbench.py` or edit `parameters.json` for controlled sensitivity tests (e.g., `OFFSET_O`, `ALPHA`/`BETA`/`GAMMA`, `RECON_RADIUS`, `MU`).

## Troubleshooting and common checks
* `symbolic_proof.txt` missing: install SymPy (`python3 -m pip install --user sympy`) and re-run with `--symbolic`.
* `width5_transitions_full.csv` has unexpected nonzero interior deltas: confirm `OFFSET_O` equals the manuscript default (2) and that the script file is unmodified.
* Large output directory: the exhaustive enumeration produces a single CSV with a finite number of rows ($5^5 = 3,125$ combinations). If you split outputs, check `ROWS_PER_PART` in the script.
* LaTeX or manuscript integration: if you include generated CSV fragments into LaTeX, escape `_`, `%`, and `&` as needed.

## Citation and license
When reporting results, include:
* author name and doi
* script name `scripts/sba_theory_proofbench.py`,
* exact command-line used, and
* the `parameters.json` produced for that run.


## Contact and contributions
**Contact:** [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com)  

---
**Final note:** This README is written so reviewers and developers can immediately test the SBA_DNA theory using the single canonical proofbench script. The script is the only program required to generate the checks and validations described in the manuscript; use it to produce deterministic artifacts for peer review, sensitivity analysis, and reproducible supplementary materials.