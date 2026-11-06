import tempfile
import unittest
import warnings
from pathlib import Path

from mlpdft.uq_calibration import resolve_calibration_file


class CalibrationSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.train, self.valid, self.test, self.cal = [root / name for name in ("train", "valid", "test", "cal")]
        for path in (self.train, self.valid, self.test, self.cal):
            path.write_text(path.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_separate_calibration(self):
        path, legacy = resolve_calibration_file(self.cal, self.train, self.valid, self.test)
        self.assertEqual(path, self.cal.resolve())
        self.assertFalse(legacy)

    def test_legacy_fallback_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path, legacy = resolve_calibration_file(None, self.train, self.valid, self.test)
        self.assertEqual(path, self.valid)
        self.assertTrue(legacy)
        self.assertEqual(len(caught), 1)

    def test_rejects_split_path_or_identical_copy(self):
        for other in (self.train, self.valid, self.test):
            with self.subTest(split=other.name):
                with self.assertRaises(ValueError):
                    resolve_calibration_file(other, self.train, self.valid, self.test)
                self.cal.write_bytes(other.read_bytes())
                with self.assertRaises(ValueError):
                    resolve_calibration_file(self.cal, self.train, self.valid, self.test)

    def test_missing_calibration_fails(self):
        self.cal.unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_calibration_file(self.cal, self.train, self.valid, self.test)


if __name__ == "__main__":
    unittest.main()
