import unittest
import dimensional_causality as dc
import numpy as np


class TestWrapper(unittest.TestCase):
    def smoke_test_infer_causality(self):
        np.random.seed(0)
        x = np.random.rand(10000)
        y = np.random.rand(10000)
        k_range = range(4,44, 1)

        probs = dc.infer_causality(x, y, 4, 1, k_range, 0.05, 3.0, 20.0, 4, False, False)
        print probs

        self.assertEqual(True, True)

    def smoke_test_infer_causality_export(self):
        np.random.seed(0)
        x = np.random.rand(10000)
        y = np.random.rand(10000)
        k_range = range(4, 44, 1)

        probs = dc.infer_causality(x, y, 4, 1, k_range, 0.05, 3.0, 20.0, 4, True)
        print probs

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
