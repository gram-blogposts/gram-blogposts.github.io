# LOF Phase Transition — Reproducibility Package

Companion code for "LOF Has a Phase Transition — and TDA Shows Where" (GRaM @ ICLR 2026).

## Quick Start

```bash
pip install -r requirements.txt
python reproduce.py        # Main results (~3 min)
python supermassive.py     # Full 22-dataset sweep (~10 min)
```

## What Gets Reproduced

- Annulus table (Table 1): LOF, DTM, iForest, ECOD, PH-Manifold, PH-Disruption
- Phase transition curve and sigmoid fit
- Breast Cancer + Wine contamination sweeps
- OpenML dataset sweep (Cardio, Optdigits, Segment)
- Cohen's d and Wilcoxon tests

## Requirements

- Python ≥ 3.10
- numpy 1.26.4, scikit-learn 1.4.2, ripser 0.6.8, scipy 1.13.0
- ~9 GB RAM recommended for n=10,000 experiments

All random seeds pinned. Expected output matches paper within ±0.005.
