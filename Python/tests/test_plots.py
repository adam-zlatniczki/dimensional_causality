import unittest
import dimensional_causality as dc


class TestPlots(unittest.TestCase):
    def test_plot_k_range_dimensions(self):
        k_range = range(5)
        exported_dims = [[2.0, 2.1, 3.0, 4.0],
                         [2.0, 2.1, 3.0, 4.0],
                         [2.0, 2.1, 3.0, 4.0],
                         [2.0, 2.1, 3.0, 4.0],
                         [2.0, 2.1, 3.0, 4.0]]
        exported_stdevs = [[0.15, 0.15, 0.15, 0.15],
                           [0.15, 0.15, 0.15, 0.15],
                           [0.15, 0.15, 0.15, 0.15],
                           [0.15, 0.15, 0.15, 0.15],
                           [0.15, 0.15, 0.15, 0.15]]
        dc.plot_k_range_dimensions(k_range, exported_dims, exported_stdevs)
        self.assertEqual(True, True)

    def test_plot_probabilities(self):
        final_probabilities = [0.15, 0.05, 0.15, 0.1, 0.55]
        dc.plot_probabilities(final_probabilities)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
