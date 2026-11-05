# QSFI–PBH

## Run a new calculation

1. Open **`QSFIPBH.nb`** in Mathematica and evaluate section **1**.
2. Evaluate the input cell in section **2** to start the complete native calculation.
3. Wait for the completion panel and **`_SUCCESS.json`** in the displayed run directory.

Initialization displays “Not started”. It does not start integration.
Run sections individually: evaluating the entire main Notebook also starts its
optional examples. 

```mathematica
runResult = Monitor[
  QSFIPBH`RunStandalone[Automatic, <|"fresh" -> True|>],
  Dynamic[QSFIPBH`RunStatusPanel[QSFIPBH`$RunState], UpdateInterval -> 1]
];
QSFIPBH`RunStatusPanel[runResult]
```

The default workflow computes **12 new coefficient rows** from native mixed
propagators and Sobol integrals, calculates extrema and PBH constraints, builds
and fits a new CMB template against the **171-bin raw observations**, and exports
six figures as PDF, SVG and PNG. The original full numerical budgets are retained;
this calculation can take a long time. Progress includes case, integration stage,
sampling and storage information.

The bundled experimental inputs are coordinates, measurements and the full
covariance, with [provenance and an upstream license](data/observations/README.md).
These observations are required measurement inputs; all theory predictions and
fitted results are calculated anew in Mathematica.

## Completion and saved results

Each invocation creates a unique directory under `runs/`. Completed cases are
checkpointed immediately. The final success marker is written **last**, after
requested computations, exports, result readback and file-integrity checks.

| File | Purpose |
| --- | --- |
| `_SUCCESS.json` | Successful completion, scope and file checksums |
| `_FAILED.json` / `_INTERRUPTED.json` | Failure or interruption, never a success claim |
| `results.wxf` | Native reloadable bundle: new coefficients, extrema, PBH/CMB results, observation arrays, settings and diagnostics |
| `coefficients.csv`, `extrema.csv` | Numerical tables |
| `pbh_constraints.json`, `cmb_fit.json` | PBH results and new CMB fit |
| `observations_used.wxf` | Actual observation arrays and provenance used |
| `config.json`, `run_manifest.json`, `progress.json`, `run.log` | Settings, inventory, status and progress log |
| `production/` | Native caches, case checkpoints, partial/final result bundles and coupling-scan WXF files |
| `cmb/`, `figures/` | New CMB template/fit and requested figure exports |

Read a successful run without repeating integration:

```mathematica
saved = QSFIPBH`LoadRunResults[runResult["output_directory"]];
saved["coefficients"]
saved["cmb_fit"]
```

`LoadRunResults` checks the success marker and saved artifact hashes before
returning the bundle. Missing or modified files produce a `Failure`. Existing
successful directories are protected from reruns. To resume an interrupted run
with its saved configuration and compatible native checkpoints:

```mathematica
QSFIPBH`RunStandalone[interruptedDirectory, <|"fresh" -> False|>]
```

A failed or aborted computation preserves completed work, records its terminal
state and creates no success marker. An operating-system or power interruption
may leave a `running` record; resume that directory to continue.

## Small independent check and custom runs

```mathematica
QSFIPBH`RunStandalone[Automatic,
  <|"test_mode" -> True, "generate_figures" -> False|>]
```

This still performs all 12 native cases and a new CMB fit, using a deliberately
small integration budget and three raw observation bins. It reports
**“Test completed (not production precision)”**, `status -> "custom_success"` and
`manuscript_usable -> False`. It is an execution check, not a convergence result.
Set `"generate_figures" -> True` to exercise all six exports too.

Other controls are `"production_overrides" -> <|...|>`,
`"cmb_overrides" -> <|...|>`, `"observations" -> jsonOrWXFPathOrAssociation`, and
`"include_cmb" -> False` for an explicitly recorded PBH-only scope. Default CMB
resolution is `"fine"`. Custom sampling/contour settings, reduced observations
and unvalidated CMB fits cannot inherit the full manuscript-budget flag.
Inspect defaults with ``QSFIPBH`ProductionConfig[]``.

## Notebook layout

| Notebook | Contents |
| --- | --- |
| `QSFIPBH.nb` | Main run, progress, reload, resume and small-run examples |
| `Notebooks/Core.nb` | Configuration, spectrum, quadrature, PBH analysis and covariance fitting |
| `Notebooks/Propagators.nb` | Mixed propagators, contours, interpolation and momentum kernels |
| `Notebooks/Projection.nb` | Native Sobol integration, errors, checkpoints, scans and new CMB fitting |
| `Notebooks/Figures.nb` | Six native figures and PDF/SVG/PNG export |
| `Notebooks/Workflow.nb` | Independent pipeline, durable storage, status and verified reload |
| `Tests.nb` | Native numerical regressions and workflow failure/integrity checks |

Keep this directory structure intact. All executable code is contained in Notebook files. `tests/oracle.json` and `reference_outputs/` are historical
**validation fixtures only**. They may be omitted for a new standalone run;
regression tests and the optional `ReproduceReference` helper need them.

