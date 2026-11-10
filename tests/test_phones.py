import unittest

from phonem_transcriptor.phones import assign_phone_times


class PhoneTimeTest(unittest.TestCase):
    def test_equal_length_keeps_the_vowel_span(self):
        spans = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 0.6)]
        timed = assign_phone_times("amor", spans, ["a", "m", "o", "ɾ"])
        self.assertEqual(timed[2], ("o", 0.2, 0.5))

    def test_digraph_keeps_both_letters(self):
        spans = [(0.0, 0.1), (0.1, 0.4)]
        timed = assign_phone_times("ch", spans, ["tʃ"])
        self.assertEqual(len(timed), 1)
        self.assertAlmostEqual(timed[0][1], 0.0)
        self.assertAlmostEqual(timed[0][2], 0.4)

    def test_one_letter_two_phones_splits_that_letter(self):
        spans = [(0.0, 1.0)]
        timed = assign_phone_times("x", spans, ["k", "s"])
        self.assertEqual(len(timed), 2)
        self.assertAlmostEqual(timed[0][1], 0.0)
        self.assertAlmostEqual(timed[0][2], 0.5)
        self.assertAlmostEqual(timed[1][1], 0.5)
        self.assertAlmostEqual(timed[1][2], 1.0)


if __name__ == "__main__":
    unittest.main()
