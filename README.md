# Specular Bit Architecture (SBA) Proofbench Repository

**Author:** Marco Oppido  
**Contact:** [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com)

## One-line summary
This repository contains the cosmological theory (primary scientific claim), the Specular Bit Architecture (SBA) formal framework derived from it (SBA‑DNA), and a single, auditable Python proofbench that reproduces, verifies, and explores the formal and modeling claims.

## Table of contents
- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Quickstart reproduction](#quickstart-reproduction)
- [Command-line usage](#command-line-usage)
- [Parameters and provenance](#parameters-and-provenance)
- [Output artifacts and interpretation](#output-artifacts-and-interpretation)
- [Recommended validation workflow](#recommended-validation-workflow-for-reviewers)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)
- [Contact and contributions](#contact-and-contributions)

---

## Overview
The cosmological manuscript proposes that spacetime and mass emerge from deterministic thermodynamic processing of quantum probability. The Specular Bit Architecture (SBA) formalizes this as an informational instruction set architecture using specular carriers (01/10), Fibonacci positional weights $W_i = F_{i+o}$, exact local rewrite rules, deterministic sanitization, and a local arbitration protocol (Protocol C).

The included proofbench (`scripts/sba_theory_proofbench.py`) is a single-file, reproducible tool that performs exhaustive local enumeration, symbolic checks, energy bookkeeping, toy continuous simulations, and unit tests to validate the formal invariants and produce modeling artifacts. The repository separates formal theorem-level claims from modeling and hypothesis-level outputs that require independent calibration.

---

## Repository layout
```text
main.tex                         # Manuscript source (LaTeX)
references.bib                   # Bibliography
scripts/
  sba_theory_proofbench.py       # Single-file proofbench (entrypoint)
figures/                         # Figures used by the manuscript
outputs/                         # Generated artifacts (CSV/TXT/JSON)
uploads/                         # Auxiliary uploaded assets
README.md                        # This file
LICENSE                          # MIT
```

---

## Quickstart reproduction

### Prerequisites
- **Python 3.8 or later**
- **Required packages:** `numpy`
- **Optional but recommended:** `sympy` (for symbolic verification)
- **Other packages:** The proofbench is mostly self-contained using the standard library.

### Exact commands (from repository root)
```bash
# 1. Install dependencies (numpy required, sympy optional for symbolic checks)
python3 -m pip install --user numpy sympy

# 2. Create outputs directory
mkdir -p outputs

# 3. Run the full proofbench pipeline
python3 scripts/sba_theory_proofbench.py --all --outdir outputs

# 4. Quick checks
wc -l outputs/width5_transitions_full.csv
less outputs/verification_report.txt
```

### Expected quick outcomes
- `outputs/width5_transitions_full.csv` — exhaustive width‑5 enumeration (3,125 rows + header).
- `outputs/verification_report.txt` — verification summary; interior-rule failures should be 0 under manuscript defaults.
- `outputs/symbolic_proof.txt` — present if SymPy is installed.
- `outputs/energy_examples.csv`, `outputs/toy_simulation.csv` — modeling artifacts.

---

## Command-line usage
```bash
python3 scripts/sba_theory_proofbench.py [OPTIONS]
```

### Main options
- `--all` : run the complete pipeline
- `--enumerate` : exhaustive width‑5 enumeration + verification reports
- `--symbolic` : symbolic Fibonacci identity check (requires SymPy)
- `--energy` : informational mass/energy examples
- `--toy` : toy continuous simulation export
- `--tests` : unit test suite
- `--prime` : prime/composite SBA experiment (exploratory)
- `--phase` : phase-map concentration test with null models
- `--outdir <path>` : output directory (default: `outputs`)

### Examples
```bash
python3 scripts/sba_theory_proofbench.py --enumerate --symbolic --tests --outdir outputs
python3 scripts/sba_theory_proofbench.py --prime --outdir outputs
```

---

## Parameters and provenance
The proofbench exposes a `Config` dataclass at the top of `scripts/sba_theory_proofbench.py`. Key parameters include:
- `OFFSET_O` (Fibonacci offset, default 2)
- `M0` (discrete bookkeeping mass unit)
- `MU` (calibration constant mapping informational mass to kg)
- `C_LIGHT` (speed of light used in $E = mc^2$)
- `ALPHA`, `BETA`, `GAMMA` (Protocol C score weights)
- `RECON_RADIUS` (reconstructor radius)

After each run, the script writes `parameters.json` into the output directory. To reproduce a run exactly:
1. Commit the repository (record commit hash).
2. Save the `parameters.json` produced by the run.
3. Re-run the proofbench with the same `--outdir` and the same code commit.

---

## Output artifacts and interpretation

### Formal / verification artifacts (theorem-level)
- `width5_transitions_full.csv` — exhaustive local transition table used to certify local rewrite classes.
- `verification_report.txt` — checks of weighted-sum preservation and boundary behavior.
- `symbolic_proof.txt` — symbolic verification of the Fibonacci identity (if SymPy installed).
- `unit_test_report.txt` — unit tests for core primitives.

### Modeling / simulation artifacts (hypothesis-level)
- `energy_examples.csv` — informational mass → physical proxy → $E = mc^2$ examples (requires calibration).
- `toy_simulation.csv` — illustrative 1D pressure model outputs.
- `prime_*` and `phase_*` outputs — exploratory experiments; interpret as statistical/modeling evidence, not formal proofs.

### Important interpretation note
Formal invariants (e.g., algebraic preservation under rewrite rules) are proved under the manuscript's synchronous semantics and Protocol C assumptions. Modeling outputs require independent calibration of constants (e.g., `MU`) before any physical claim.

---

## Recommended validation workflow (for reviewers)
1. **Reproduce canonical run:** Run `--all` with default parameters; confirm `verification_report.txt` shows zero interior failures.
2. **Symbolic check:** Install SymPy and run `--symbolic`; inspect `symbolic_proof.txt`.
3. **Parameter sensitivity:** Modify `OFFSET_O`, `ALPHA/BETA/GAMMA`, and `RECON_RADIUS` in the config to test computational robustness.
4. **Energy mapping sanity:** Inspect `energy_examples.csv` and confirm units and orders of magnitude given your chosen `MU`.
5. **Determinism:** Re-run the same command twice and compare SHA256 hashes of generated artifacts.

---

## Troubleshooting
- **SymPy or Numpy not installed:** install with `python3 -m pip install --user numpy sympy`.
- **CSV column mismatch:** ensure the proofbench completed successfully; re-run `--enumerate`.
- **LaTeX compile issues:** check `main.tex` for unescaped characters if you include generated fragments.
- **Interior verification failures:** confirm `OFFSET_O` matches the manuscript and that the proofbench file logic is unchanged.

---

## Citation and license
Suggested citation elements: Author, manuscript title, repository name, commit hash or release tag, date of access.  

---

## Contact
Contact: [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com).  

### Final note
This repository bundles a formal informational framework (SBA), a cosmological interpretation that motivates it, and a single, auditable proofbench for reproducible verification and exploration. Use the proofbench to generate artifacts for review, to perform sensitivity analyses, and to prepare reproducible supplementary materials for publication.
