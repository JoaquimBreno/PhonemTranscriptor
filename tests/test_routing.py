import tempfile
import unittest
from pathlib import Path

from phonem_transcriptor.paths import sofa_checkpoint, sofa_dictionary
from phonem_transcriptor.routing import aligner_for


class RoutingTest(unittest.TestCase):
    def test_sofa_when_assets_exist_and_mms_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for language in ("en", "zh", "ko"):
                sofa_checkpoint(root, language).parent.mkdir(parents=True)
                sofa_checkpoint(root, language).write_bytes(b"ckpt")
                sofa_dictionary(root, language).write_text("a\ta\n", encoding="utf-8")
                self.assertEqual(aligner_for(language, root), "sofa")
            self.assertEqual(aligner_for("pt", root), "mms")
            self.assertEqual(aligner_for("es", root), "mms")
            self.assertEqual(aligner_for("fr", root), "mms")

    def test_missing_sofa_asset_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                aligner_for("en", Path(tmp))
            with self.assertRaises(FileNotFoundError):
                aligner_for("zh", Path(tmp))
            with self.assertRaises(FileNotFoundError):
                aligner_for("ko", Path(tmp))


if __name__ == "__main__":
    unittest.main()
