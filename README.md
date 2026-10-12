# QSFI–PBH manuscript calculations

This repository contains the Python code used to compute the quasi-single-field
inflation (QSFI) non-Gaussianity projections and the corresponding primordial
black hole (PBH) abundance results discussed in the manuscript. The production
workflow evaluates the bispectrum, contact trispectrum, and scalar-exchange
trispectrum, then exports the numerical tables and three manuscript figures.
The small files in `reference_outputs/` are comparison outputs, not runtime
inputs.

## Requirements and installation

Use Python 3.12 (tested); the package also declares support for Python 3.11.
From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`pyproject.toml` is the single source of truth for Python dependencies.
`requirements.txt` contains only `-e .`, so
`python -m pip install -r requirements.txt` is an equivalent installation
command. Do not run both installation commands.

## Production configuration

The defaults in `qsfi_pbh/config.py` are:

| Parameter | Value |
| --- | --- |
| Mixing coupling `lambda_2/H` | `0.38` |
| Cubic coupling `lambda_3/H` | `[-1, 1]`, 2001 points |
| Quartic coupling `lambda_4` | `[-1, 1]`, 2001 points |
| Sound-speed ratios `r_s=c_phi/c_chi` | `0.3, 1, 3` |
| Heavy-field mass parameters `nu_tilde` | `0.1, 0.3, 0.5, 1.0` |
| Target PBH abundance | `10^-15` |
| Reference peak height | `0.01` |
| Non-Gaussianity bound | `Q_NG=S_3+S_4/8 <= 6` |

The `lambda_3/H` and `lambda_4` arrays are each spaced by `0.001`.
The sound-speed pairs `(c_phi,c_chi)` are `(0.3,1)`, `(1,1)`, and
`(1,1/3)`.
The metadata contract retains an `update17` identifier for compatibility with
the manuscript datasets; it is a version label, not a filesystem dependency.

## Reproduce the manuscript results

The main command computes all production datasets, writes the numerical
report, and renders the three figures:

```bash
python reproduce_all.py
```

`compute_results.py` performs the same computation and rendering directly.
After a successful production run, redraw the figures without repeating the
integrals using either:

```bash
python reproduce_all.py --figures-only
```

or:

```bash
python make_figures.py
```

The figure-only commands require the four generated NPZ datasets in
`figure_data/manuscript/`. They fail with an explanatory metadata error if
the data are missing or come from an incompatible configuration.

For a completely new, non-resuming computation, run:

```bash
python reproduce_fresh.py
```

This creates a new timestamped directory under `fresh_runs/` with its own
empty cache, datasets, figures, Matplotlib configuration, and audit manifest.
It never reads the normal `.cache/`, `figure_data/`, or `figures/` trees.
An existing run directory is refused. To choose an unused destination, pass
`--run-dir /path/to/a/new/directory`.
The fresh launcher and output audit are covered by tests; a complete cold
production run was not repeated during release packaging because of its cost.

Full production evaluates expensive momentum-space integrals and may require
hours or longer, depending on the machine. The original run used more than
2 GB of generated cache and data. The 2001-by-2001 grids are assembled for
12 sound-speed and mass panels. The ordinary production command checkpoints
completed integrations and can resume from a matching local cache; the fresh
command always starts with an empty cache.

## Output layout

```text
.
├── qsfi_pbh/                    # Physical model and numerical workflow
├── compute_results.py            # Compute data, report, and figures
├── make_figures.py               # Render figures from production NPZ data
├── reproduce_all.py              # Main reproduction command
├── reproduce_fresh.py            # New empty-cache run with output audit
├── check_exchange_convergence.py # Optional scalar-exchange diagnostic
├── tests/                        # Unit and workflow tests
├── reference_outputs/            # Small, tracked manuscript comparison outputs
│   ├── manuscript_numerical_results.txt
│   └── figures/                   # Three final PDF figures
├── figure_data/                  # Generated; ignored by Git
├── figures/                      # Generated; ignored by Git
├── .cache/                       # Generated checkpoints; ignored by Git
└── fresh_runs/                   # Generated isolated runs; ignored by Git
```

Production creates four pickle-free NPZ datasets in
`figure_data/manuscript/`: `hatted_cumulant_coefficients.npz`,
`r3_cumulant_contributions.npz`,
`sound_speed_log10_beta_ratio_full.npz`, and
`sound_speed_comparison_full.npz`. The numerical report is
`figure_data/manuscript_numerical_results.txt`. The three figure stems are
`r3_relative_contribution_bars`,
`sound_speed_log10_beta_ratio_full`, and
`sound_speed_comparison_full`; each is exported as PDF, SVG, and PNG.
