# Historical validation fixtures

These files are optional for a new standalone calculation. `RunStandalone`
does not read this directory. Numerical regressions and the optional
`ReproduceReference` helper use the archived material for comparison.

- `manuscript_numerical_results.txt`: corrected report; four-point coefficients
  have already been corrected and must not be reweighted again.
- `data/`: corrected coefficient/extremum tables and a PBH constraint example.
- `inputs/archived_numerical_results.txt`: original pre-correction report.
  The archive reader applies the inverse-cube sound-speed correction exactly once.
- `inputs/cmb_fit_fine.json`, `inputs/cmb_template_fine.json`: archived computed
  fit and template, not the raw observational inputs.
- `inputs/provenance.json`: historical input hashes and method.

Reference rendering does not constitute a new cold integration. Native fonts,
PDF metadata and adaptive plotting may differ; regression comparisons use
numerical tolerances rather than identical output bytes.
