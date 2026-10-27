
import unittest
from pathlib import Path

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_artifact_exists(self):
        self.assertTrue(
            Path("student_result_model.pkl").exists(),
            "Model artifact was not created"
        )

    def test_metrics_file_exists(self):
        self.assertTrue(
            Path("metrics.json").exists(),
            "Metrics file was not created"
        )

    def test_model_can_be_loaded(self):
        model = joblib.load("student_result_model.pkl")
        self.assertIsNotNone(model)

    def test_model_can_make_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        }])

        prediction = model.predict(sample)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
