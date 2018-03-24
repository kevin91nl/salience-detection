import json
import unittest

from utils.validators import is_valid_esd_json


class TestValidators(unittest.TestCase):

    def test_is_valid_esd_json(self):
        # It is a valid non-train/test document
        doc = {"text": "A text."}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=False))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # The next document is not a valid train/test document since it contains no entities field
        doc = {"text": "A text.", "abstract": "Text."}
        with self.assertRaises(ValueError) as _:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))

        # This document is not valid since it contains no "text" field
        doc = {}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=False)

        # This train/test document is not valid, since the "entities" are not a list
        doc = {"text": "A text.", "abstract": "Text.", "entities": "Not a list."}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

        # This is a valid train/test document
        doc = {"text": "A text.", "abstract": "Text.", "entities": []}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # This is not a valid train/test document since it has no abstract
        doc = {"text": "A text.", "entities": []}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

        # This is a valid train/test document
        doc = {"text": "A text.", "abstract": "Text.", "entities": [{"entity": "Entity", "salience": True}]}
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        # This is not a valid document, since one entity has no "salience" field
        doc = {"text": "A text.", "abstract": "Text.", "entities": [{"entity": "Entity"}]}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

        # This is not a valid document, since one entity has no "entity" field
        doc = {"text": "A text.", "abstract": "Text.", "entities": [{"salience": False}]}
        with self.assertRaises(ValueError) as _:
            is_valid_esd_json(doc, is_train_document=True)

    def test_is_valid_esd_json_readme_example(self):
        # Test the examples in the README file
        doc_str = """{
            "text": "This text is about dogs. Dogs are animals. Cats are also animals.",
            "abstract": "Text about dogs.",
            "entities": [
                {
                    "entity": "dogs",
                    "salience": true
                },
                {
                    "entity": "cats",
                    "salience": false
                }
            ]
        }"""
        doc = json.loads(doc_str)
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=True))
        except ValueError:
            self.fail('A ValueError was raised but should not.')

        doc_str = """{
          "text": "This text is about dogs. Dogs are animals. Cats are also animals."
        }"""
        doc = json.loads(doc_str)
        try:
            self.assertEqual(True, is_valid_esd_json(doc, is_train_document=False))
        except ValueError:
            self.fail('A ValueError was raised but should not.')
