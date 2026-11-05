# Planck PR4 observational input

`planck_pr4_binned_shape.json` contains the 171 measured binned primordial bispectrum shape values, their full 171×171 covariance, and triangle-bin coordinates `x=k1/k3`, `y=k2/k3`. These are observation-derived data and a simulation-calibrated covariance, supplied independently of this project's QSFI theory calculations.

The four arrays come from [`binned_shape_data.npz`](https://github.com/oliverphilcox/Binned-Bispectrum-Shape/blob/main/binned_shape_data.npz) in Oliver H. E. Philcox's [Binned-Bispectrum-Shape repository](https://github.com/oliverphilcox/Binned-Bispectrum-Shape). All binary64 entries are preserved exactly by decimal round-trip. Mathematica reads the JSON directly.

`provenance.json` records source and JSON SHA256 hashes, array dimensions and exact round-trip checks. `LICENSE` preserves the upstream MIT copyright and permission notice.

Please cite: Oliver H. E. Philcox, *What Shape is the Inflationary Bispectrum?* (2026), [arXiv:2603.17004](https://arxiv.org/abs/2603.17004).

No archived QSFI coefficients, computed QSFI templates or fitted QSFI amplitudes are included in this folder.

The covariance's two triangles differ by at most `0.015625`, or `7.40e-18` relative to its largest entry, due to binary64 roundoff. The JSON preserves both triangles exactly. Native fitting checks symmetry relative to the data scale and uses the symmetric average only in its numerical working copy before Cholesky factorization.
