from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from make_figures import EXPECTED_FIGURE_PATHS
from qsfi_pbh.config import ProductionConfig
from qsfi_pbh.figure_data import EXPECTED_DATA_PATHS, write_figure_data
from qsfi_pbh.reporting import REPORT_RELATIVE_PATH
from reproduce_fresh import FreshRunPaths, audit_fresh_outputs, run_fresh


class FreshAuditTest(unittest.TestCase):
    def _complete_outputs(self, paths: FreshRunPaths, *, report: str) -> None:
        metadata = ProductionConfig().metadata()
        for relative_path in EXPECTED_DATA_PATHS:
            write_figure_data(
                paths.output_root / relative_path,
                metadata={**metadata, "kind": relative_path.stem},
                arrays={"sample": [1.0]},
            )
        report_path = paths.output_root / REPORT_RELATIVE_PATH
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        for relative_path in EXPECTED_FIGURE_PATHS:
            destination = paths.figure_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"figure")

    def test_audit_accepts_exact_data_report_and_figure_contract(self) -> None:
        with TemporaryDirectory() as directory:
            paths = FreshRunPaths.for_run_dir(Path(directory))
            paths.output_root.mkdir()
            paths.figure_root.mkdir()
            self._complete_outputs(
                paths,
                report="[fixed_power_extrema]\n[fixed_abundance_extrema]\n[projected_coefficients]\n",
            )

            result = audit_fresh_outputs(paths)

        self.assertEqual(result["data_files"], 4)
        self.assertEqual(result["report_files"], 1)
        self.assertEqual(result["figure_files"], 9)

    def test_audit_rejects_report_missing_required_heading(self) -> None:
        with TemporaryDirectory() as directory:
            paths = FreshRunPaths.for_run_dir(Path(directory))
            paths.output_root.mkdir()
            paths.figure_root.mkdir()
            self._complete_outputs(paths, report="[fixed_power_extrema]\n")

            with self.assertRaisesRegex(RuntimeError, "report.*heading"):
                audit_fresh_outputs(paths)

    def test_fresh_launcher_uses_only_new_scientific_roots(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "package"
            package.mkdir()
            old_cache = package / ".cache"
            old_cache.mkdir()
            (old_cache / "sentinel").write_text("old", encoding="utf-8")
            run_dir = root / "new-run"
            observed = {}

            def fake_runner(command, *, check, cwd, env):
                paths = FreshRunPaths.for_run_dir(run_dir)
                observed["command"] = command
                observed["cwd"] = cwd
                observed["env"] = env
                self.assertTrue(check)
                self.assertEqual(cwd, package)
                self.assertEqual(command[0], "python-test")
                self.assertEqual(command[1], str(package / "compute_results.py"))
                self.assertEqual(command[2:], (
                    "--cache-root", str(paths.cache_root),
                    "--output-root", str(paths.output_root),
                    "--figure-root", str(paths.figure_root),
                ))
                self.assertEqual(env["MPLCONFIGDIR"], str(paths.mplconfig_root))
                self.assertTrue(all(not any(path.iterdir()) for path in (
                    paths.cache_root, paths.output_root, paths.figure_root,
                )))

            result = run_fresh(
                run_dir,
                package_dir=package,
                python_executable="python-test",
                runner=fake_runner,
                auditor=lambda paths: {"checked": True},
            )

            self.assertEqual(result.run_dir, run_dir)
            self.assertEqual((old_cache / "sentinel").read_text(), "old")
            self.assertTrue(observed)
            with self.assertRaises(FileExistsError):
                run_fresh(
                    run_dir,
                    package_dir=package,
                    python_executable="python-test",
                    runner=fake_runner,
                    auditor=lambda paths: {"checked": True},
                )


if __name__ == "__main__":
    unittest.main()
