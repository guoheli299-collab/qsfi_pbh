# Raw observational input

`observations/` contains the released triangle coordinates, measured binned
bispectrum and full covariance. These experimental inputs are sufficient for
the native CMB workflow; Mathematica computes the theoretical template and
fits it anew. See `observations/README.md` and `provenance.json` for source,
array hashes and the separate upstream license.

Archived calculated coefficients, templates and fits have been moved to
`../reference_outputs/inputs/`. They are historical validation fixtures and
are never loaded by `RunStandalone`.
