import unittest

import numpy as np

from dataset.converter import convert_rsd_batch


class TestRSDConverter(unittest.TestCase):

    def test_converter(self):
        batch = [{'features': [1, 2, 3], 'is_relevant': 0}, {'features': [4, 5], 'is_relevant': 1}]
        result = convert_rsd_batch(batch)
        self.assertTrue('xs' in result, 'The converted batch should have a "xs" field.')
        self.assertTrue('ts' in result, 'The converted batch should have a "ts" field.')
        self.assertEqual([[1, 2, 3], [4, 5]], result['xs'])
        self.assertTrue(issubclass(result['ts'].dtype.type, np.integer), 'The "ts" field should be an integer type.')
        self.assertEqual((2,), result['ts'].shape, 'The "ts" field should consist of 2 elements.')
        self.assertEqual(0, result['ts'][0])
        self.assertEqual(1, result['ts'][1])

    def test_invalid_device(self):
        with self.assertRaises(NotImplementedError):
            convert_rsd_batch([], 'Invalid device.')
