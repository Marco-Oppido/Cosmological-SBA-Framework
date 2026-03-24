# Specular Bit Architecture (SBA) — Cosmological Mapping and Proofbench

**Author:** Marco Oppido  
**Contact:** [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com)  
**Paper & Foundation:** Published on Zenodo: [![DOI]https://doi.org/10.5281/zenodo.19201218]
                                             [![DOI]https://doi.org/10.5281/zenodo.19181895]

## Overview
This repository contains the cosmological theory (the principal scientific contribution), the Specular Bit Architecture (SBA) formal framework derived from that theory (SBA‑DNA), and a single, comprehensive Python proofbench that reproduces, verifies, and exercises the formal claims numerically and symbolically.

**Core idea:** The cosmological manuscript proposes that spacetime, mass, and macroscopic reality emerge from a deterministic thermodynamic processing of quantum probability. The Specular Bit Architecture is the formal informational ISA that implements this processing: specular carriers (01/10), Fibonacci positional weights (W_i = F_{i+o}), exact local rewrite rules, deterministic sanitization, and a local arbitration protocol (Protocol C). SBA‑DNA formalizes the local rules and invariants; the proofbench implements exhaustive enumeration, symbolic checks, energy bookkeeping, a toy continuous model, and unit tests to validate the claims.

This README explains the repository contents, how to reproduce the verification artifacts, how to interpret the outputs, and recommended validation steps for reviewers.

## Repository layout

```text
Code
SBA_paper.tex                 # Main LaTeX manuscript (cosmology + SBA formalism)
references.bib                # Bibliography
scripts/
  sba_theory_proofbench.py    # Single-file Python proofbench (enumeration, verification, simulation)
figures/                      # Figures used by the manuscript
outputs/                      # Generated artifacts (CSV, TXT, reports) — created by the proofbench
README.md                     # This file
LICENSE                       # MIT License
```

## What is the proofbench
`scripts/sba_theory_proofbench.py` is a single, self-contained Python program that:
- Computes Fibonacci positional weights (supports negative indices via `F_{−n} = (−1)^{n+1} F_n`).
- Exhaustively enumerates all width‑5 neighborhoods in `{−2,−1,0,1,2}^5`.
- Applies interior and boundary rewrite rules and verifies algebraic invariants (weighted‑sum preservation).
- Simulates deterministic sanitization and Protocol C arbitration with a deterministic reconstructor.
- Produces canonical artifacts: `width5_transitions_full.csv`, `verification_report.txt`, `symbolic_proof.txt` (if SymPy is installed), `energy_examples.csv`, `toy_simulation.csv`, and unit‑test reports.
- Provides example mappings from informational mass to a physical proxy and evaluates `E = mc^2` for sample configurations.
- Includes a toy 1D continuous model that maps a probability density `ρ` to a pressure `P(ρ)` and computes resulting accelerations (illustrative).

The proofbench is intentionally deterministic and designed to be auditable by reviewers.

## Quick reproduction (exact commands)
Run these commands from the repository root. They are minimal and deterministic.

```bash
# 1. Ensure Python 3.8+ is available
python3 --version

# 2. (Optional) install SymPy for symbolic verification
python3 -m pip install --user sympy

# 3. Create outputs directory
mkdir -p outputs

# 4. Run the single-file proofbench (generates CSV, TXT, and verification artifacts)
python3 scripts/sba_theory_proofbench.py --all --outdir outputs

# 5. Inspect key artifacts
wc -l outputs/width5_transitions_full.csv
less outputs/verification_report.txt

# 6. Compile the LaTeX manuscript (from repository root)
pdflatex SBA_paper.tex
bibtex SBA_paper || true
pdflatex SBA_paper.tex
pdflatex SBA_paper.tex
```

## Expected results
- `outputs/width5_transitions_full.csv` — exhaustive width‑5 table (1 header + 3,125 rows).
- `outputs/verification_report.txt` — numeric verification summary and counts of any failures.
- `outputs/symbolic_proof.txt` — SymPy verification of the Fibonacci identity (if SymPy installed).
- `outputs/energy_examples.csv` — sample mappings of informational mass to physical proxy and energy.
- `outputs/toy_simulation.csv` — toy continuous model output.

The compiled PDF should include Appendix B fragments if you include or generate them.

## Parameters and where to change them
Key parameters are defined in the `Config` dataclass at the top of `scripts/sba_theory_proofbench.py`. Important parameters include:
- `OFFSET_O` — Fibonacci offset `o`. Default: 2 (admissible Zeckendorf regime).
- `M0` — discrete bookkeeping mass unit (dimensionless).
- `MU` — calibration constant (kg per unit weight) mapping dimensionless informational mass to a physical mass proxy.
- `C_LIGHT` — speed of light used in `E=mc^2`.
- `ALPHA`, `BETA`, `GAMMA` — Protocol C score weights.
- `RECON_RADIUS` — reconstruction radius `r`.
- `PRESSURE_KAPPA`, `PRESSURE_EXPONENT` — toy continuous model parameters.

To explore sensitivity, edit these values in the `Config` dataclass or modify the generated `parameters.json` and re-run the proofbench.

## What to check in the outputs

### CSV integrity
- `wc -l outputs/width5_transitions_full.csv` → should report 3126 lines (1 header + 3,125 rows).
- Check column counts: `awk -F',' '{print NF}' outputs/width5_transitions_full.csv | sort -n | uniq -c` should return a single NF value.

### Interior rule invariant
- Open `outputs/verification_report.txt` and confirm "Interior rule failures: 0" (when using manuscript defaults).

### Symbolic identity
- If SymPy is installed, `outputs/symbolic_proof.txt` should contain the symbolic simplification showing the identity `2W_i = W_{i+1} + W_{i−2}`.

### Energy bookkeeping
- Inspect `outputs/energy_examples.csv` to see how `m_sba` maps to `m_phys` and `E` for sample configurations.

### Determinism
- Re-run the proofbench twice and compare SHA256 hashes of generated files to confirm byte‑for‑byte reproducibility.

## Recommended validation workflow for reviewers
1. Reproduce canonical run with default parameters and confirm the verification report shows zero interior failures.
2. Run symbolic check (install SymPy) and confirm the algebraic identity is verified.
3. Perform sensitivity studies: vary `OFFSET_O`, `MU`, `ALPHA/BETA/GAMMA`, and `RECON_RADIUS` and observe how verification counts and energy examples change.
4. Inspect toy model to understand how a chosen pressure law maps to effective accelerations (illustrative bridge to cosmological interpretation).
5. If including outputs in the submission, mark them as generated artifacts and provide the `parameters.json` used to create them.

## Troubleshooting
- **SymPy not installed:** symbolic proof will be skipped. Install with `python3 -m pip install --user sympy`.
- **CSV column mismatch:** re-run the proofbench and inspect the first problematic rows; ensure the script completed without interruption.
- **LaTeX compile errors:** check for unescaped special characters in generated fragments; escape `_` → `\_` and `%` → `\%` if needed.
- **Interior verification failures:** confirm `OFFSET_O` in the proofbench matches the manuscript; inspect `outputs/verification_report.txt` for sample failing rows.
- **Non-deterministic outputs:** ensure the proofbench file is unchanged between runs and that the environment is stable; the proofbench is deterministic by design.

## Scientific and interpretative notes
The cosmological manuscript is the primary scientific claim; SBA‑DNA is the formal informational framework derived from it. The proofbench is a reproducible verification and exploration tool, not a substitute for experimental calibration.
- **Dimensional caution:** Fibonacci weights are dimensionless integers. The mapping to physical mass requires a calibration constant `μ` (kg per unit weight). Any physical interpretation or experimental claim requires independent calibration and validation.
- **Annihilation semantics:** the manuscript and proofbench adopt a model-level rule for cancellation and energy release; the physical interpretation of emitted radiation is substrate-dependent and must be treated as a modeling hypothesis.

## Citation and license

### Suggested citation
Marco Oppido, Specular Bit Architecture and Cosmological Mapping: Formal framework and verification artifacts, 2026 (repository).

### License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contact and contributions
For collaborations: [scaccomarco@gmail.com](mailto:scaccomarco@gmail.com).

## Final remark
This repository bundles a formal informational framework (SBA), a cosmological interpretation that motivates it, and a single, auditable proofbench that reproduces the formal artifacts. Use the proofbench to reproduce results, test alternative parameterizations, and prepare artifacts for peer review.

[def]: https://doi.org/10.5281/zenodo.19181895
