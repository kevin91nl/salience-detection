import unittest

import numpy as np

from dataset.converter import convert_rsd_batch, convert_entmem_batch


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


class TestEntMemConverter(unittest.TestCase):

    def test_converter(self):
        batch = [{'features': [[[1, 2, 3]]], 'is_salient': 0, 'query_entity_features': [1, 1, 1],
                  'context_entities_features': [1, 1, 1], 'postag_ids': [1, 2, 3], 'word_ids': [1, 2, 3],
                  'words': [1, 2, 3], 'postags': [1, 2, 3]}]
        result = convert_entmem_batch(batch)
        assert 'idx_postags' in result
        assert 'idx_words' in result
        assert 'txt_words' in result
        assert 'txt_postags' in result
        assert 'is_salient' in result

    def test_invalid_device(self):
        with self.assertRaises(NotImplementedError):
            convert_entmem_batch([], 'Invalid device.')
