import json
import tempfile
import unittest
from pathlib import Path

from phonem_transcriptor.game_lang import game_language_id, load_lang_map
from phonem_transcriptor.paths import game_model_dir


class GameLanguageTest(unittest.TestCase):
    def test_known_codes_use_the_map_and_the_rest_stay_universal(self):
        mapping = {"en": 1, "ja": 2, "yue": 3, "zh": 4}
        self.assertEqual(game_language_id("en", mapping), 1)
        self.assertEqual(game_language_id("ZH", mapping), 4)
        self.assertEqual(game_language_id("yue", mapping), 3)
        self.assertEqual(game_language_id("pt", mapping), 0)
        self.assertEqual(game_language_id("es", mapping), 0)
        self.assertEqual(game_language_id("ko", mapping), 0)
        self.assertEqual(game_language_id(None, mapping), 0)

    def test_loader_rejects_an_id_the_embedding_cannot_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lang_map.json"
            path.write_text(json.dumps({"en": 200}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_lang_map(Path(tmp))

    def test_installed_checkpoint_keeps_portuguese_on_zero(self):
        model = game_model_dir()
        if not (model / "lang_map.json").is_file():
            self.skipTest("GAME checkpoint is not installed")
        mapping = load_lang_map(model)
        self.assertEqual(game_language_id("pt", mapping), 0)
        self.assertEqual(game_language_id("en", mapping), mapping["en"])
        self.assertEqual(game_language_id("zh", mapping), mapping["zh"])


if __name__ == "__main__":
    unittest.main()
