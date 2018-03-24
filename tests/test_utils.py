import unittest

from utils import compute_jaccard_index, is_valid_esd_json


class TestUtils(unittest.TestCase):

    def test_compute_jaccard_index(self):
        # Expected value test
        set1 = {'a', 'b', 'c'}
        set2 = {'a', 'b', 'd'}
        self.assertEqual(2. / (3. + 3. - 2.), compute_jaccard_index(set1, set2))

        # Boundary test
        set1 = {'a'}
        set2 = set()
        self.assertEqual(0., compute_jaccard_index(set1, set2))

        # Error test
        with self.assertRaises(ValueError) as _:
            compute_jaccard_index(set(), set())

    def test_is_valid_esd_json(self):
        # It is a valid non train/test document
        doc = {"text": "A text."}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=False))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # It is not a valid train/test document since it contains no entities
        with self.assertRaises(ValueError) as _:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))

        # This document is not valid since it contains no "text" field
        doc = {}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=False)

        # This train/test document is not valid, since the "entities" are not a list
        doc = {"text": "A text.", "entities": "Not a list."}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

        # This is a valid train/test document
        doc = {"text": "A text.", "entities": []}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # This is a valid train/test document
        doc = {"text": "A text.", "entities": [{"entity": "Entity", "salience": True}]}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # This is not a valid document, since one entity has no "salience" field
        doc = {"text": "A text.", "entities": [{"entity": "Entity"}]}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

        # This is not a valid document, since one entity has no "entity" field
        doc = {"text": "A text.", "entities": [{"salience": False}]}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)


if __name__ == '__main__':
    unittest.main()
