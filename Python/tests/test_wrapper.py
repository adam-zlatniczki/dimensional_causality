import unittest
import dimensional_causality as dc
from dimensional_causality.util import embed
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

    def smoke_test_infer_causality_from_manifolds(self):
        np.random.seed(0)
        x = np.random.rand(1000)
        y = np.random.rand(1000)

        X = embed(x, 3, 1)
        Y = embed(x, 3, 1)
        J = X + Y
        Z = X + np.random.permutation(Y)

        k_range = range(4, 44, 1)

        print X.shape
        probs = dc.infer_causality_from_manifolds(X, Y, J, Z, k_range)
        print probs

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
