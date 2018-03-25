import unittest

import numpy as np

from model.relevant_sentence import RSD


class TestRSDModel(unittest.TestCase):

    def test_functions(self):
        # Just set some random parameters
        model = RSD(3, 5, 7)

        # Construct input of batch size 3
        xs = [[1], [0], [2]]
        ts = np.asarray([0, 1, 0], dtype='i')

        # The loss should be reduced and should be a single number (shape=())
        self.assertEqual((), model(xs, ts).shape)

        # The predictions should form a batch_size x 2 matrix and the rows should add up to 1.0
        scores = model.predict(xs).data
        self.assertEqual(scores.shape, (3, 2))
        row_sums = np.sum(scores, axis=1)
        for row_sum in row_sums:
            self.assertEqual(1.0, row_sum)
