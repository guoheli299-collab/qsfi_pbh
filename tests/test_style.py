from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qsfi_pbh import style


class _StubFigure:
    def savefig(self, destination: Path, **_: object) -> None:
        Path(destination).write_bytes(b"figure")


class PublicationFigureExportTest(unittest.TestCase):
    def test_temporary_export_name_is_not_hidden_in_finder(self) -> None:
        prefixes: list[str] = []
        real_named_temporary_file = tempfile.NamedTemporaryFile

        def capture_prefix(*args: object, **kwargs: object):
            prefixes.append(str(kwargs["prefix"]))
            return real_named_temporary_file(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                style.tempfile,
                "NamedTemporaryFile",
                side_effect=capture_prefix,
            ):
                style.save_publication_figure(
                    _StubFigure(),
                    Path(directory) / "figure",
                    formats=("png",),
                )

        self.assertEqual(len(prefixes), 1)
        self.assertFalse(prefixes[0].startswith("."))


if __name__ == "__main__":
    unittest.main()
