import unittest
from project import EVAL, TRAIN, demo


class ModelReportTests(unittest.TestCase):
    def test_disjoint_text_and_threshold_coverage(self):
        self.assertFalse(
            {row[2] for row in EVAL} & {text for rows in TRAIN.values() for text in rows}
        )
        output = demo()
        for selector in ("baseline", "candidate"):
            result = output[selector]
            self.assertEqual(sum(map(sum, result["confusion"])), len(EVAL))
            coverage = [entry["coverage"] for entry in result["thresholds"].values()]
            self.assertEqual(coverage, sorted(coverage, reverse=True))
        self.assertTrue(output["changed_predictions"])
