"""The figure cache respects an isolated production environment."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qsfi_pbh import backend


class MatplotlibCacheTests(unittest.TestCase):
    def test_explicit_mplconfig_does_not_create_project_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            isolated = root / "fresh" / "mplconfig"
            with patch.object(backend, "PROJECT_DIR", root):
                with patch.dict(os.environ, {"MPLCONFIGDIR": str(isolated)}):
                    backend.configure_matplotlib_cache()
                    self.assertEqual(os.environ["MPLCONFIGDIR"], str(isolated))
            self.assertTrue(isolated.is_dir())
            self.assertFalse((root / ".cache").exists())

    def test_default_mplconfig_stays_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(backend, "PROJECT_DIR", root):
                with patch.dict(os.environ, clear=True):
                    backend.configure_matplotlib_cache()
                    self.assertEqual(
                        os.environ["MPLCONFIGDIR"],
                        str(root / ".cache" / "mplconfig"),
                    )
            self.assertTrue((root / ".cache" / "mplconfig").is_dir())


if __name__ == "__main__":
    unittest.main()
