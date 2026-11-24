import unittest
from src.meter import Meter, pattern_to_matra, prastara, valid_patterns, classify_syllables
from src.decoder import constrained_decode, unconstrained_decode
from src.evaluation import evaluate


class ProjectTests(unittest.TestCase):
    def test_prastara_cardinality(self):
        self.assertEqual(len(prastara(5)), 32)
        self.assertEqual(prastara(2), ["LL", "LG", "GL", "GG"])

    def test_meter_filter(self):
        self.assertTrue(all(pattern_to_matra(p) == 4 for p in valid_patterns(3, 4)))

    def test_classification_is_nonempty(self):
        self.assertTrue(classify_syllables("dawn lights the river"))

    def test_constraint(self):
        meter = Meter("LGGL")
        poem = constrained_decode(meter, lines=10)
        self.assertEqual(evaluate(poem, meter)["meter_validity"], 1.0)

    def test_baseline_is_not_forced(self):
        meter = Meter("LGGL")
        poem = unconstrained_decode(lines=10)
        self.assertLess(evaluate(poem, meter)["meter_validity"], 1.0)


if __name__ == "__main__":
    unittest.main()
