from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from qsfi_pbh import backend


class DirectMixingCacheTest(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        args: Namespace,
        *,
        lambda2_over_H: float,
    ) -> tuple[Path, Path]:
        summary_path = root / "summary.json"
        data_path = root / "data.npz"
        metadata = backend.expected_metadata(args)
        metadata["lambda2_over_H"] = lambda2_over_H
        summary_path.write_text(
            json.dumps({"metadata": metadata}),
            encoding="utf-8",
        )
        nus = np.asarray(args.nus, dtype=float)
        axis = np.linspace(
            args.lambda3_min, args.lambda3_max, int(args.nscan), dtype=float
        )
        shape = (nus.size, axis.size, axis.size)
        np.savez_compressed(
            data_path,
            nus=nus,
            lambda3_over_H=axis,
            lambda4=axis,
            S4=np.zeros(shape),
            QNG=np.zeros(shape),
        )
        return summary_path, data_path

    def test_cache_requires_exact_lambda2_metadata(self) -> None:
        args = backend.make_args()
        args.nus = [0.1]
        args.nscan = 3

        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, data_path = self._write_cache(
                root,
                args,
                lambda2_over_H=args.lambda2_over_H,
            )
            with (
                patch.object(backend, "SUMMARY_PATH", summary_path),
                patch.object(backend, "DATA_PATH", data_path),
            ):
                self.assertTrue(backend.cached_cumulant_data_is_current(args))

            summary_path, data_path = self._write_cache(
                root,
                args,
                lambda2_over_H=0.8,
            )
            with (
                patch.object(backend, "SUMMARY_PATH", summary_path),
                patch.object(backend, "DATA_PATH", data_path),
            ):
                self.assertFalse(backend.cached_cumulant_data_is_current(args))


if __name__ == "__main__":
    unittest.main()
