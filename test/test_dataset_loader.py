import os
import shutil
import unittest

import nltk

from dataset.loader import RelevantSentencesLoader, EntitySalienceLoader


class TestLoaders(unittest.TestCase):

    def setUp(self):
        # Test whether the resources folder exists
        self.resources_path = os.path.join(os.path.dirname(__file__), 'resources')
        self.json_path = os.path.join(self.resources_path, 'json_examples')
        assert os.path.exists(self.resources_path) and os.path.isdir(self.resources_path)
        assert os.path.exists(self.json_path) and os.path.isdir(self.json_path)


class TestRelevantSentencesLoader(TestLoaders):

    def setUp(self):
        super(TestRelevantSentencesLoader, self).setUp()
        nltk.download('punkt')

    def test_nonexisting_path(self):
        with self.assertRaises(IOError):
            RelevantSentencesLoader(os.path.join(self.resources_path, 'nonexisting_directory'), nltk.sent_tokenize,
                                    lambda sent: sent)

    def test_max_files(self):
        loader = RelevantSentencesLoader(self.json_path, nltk.sent_tokenize, lambda sent: sent, max_files=0)
        self.assertEqual(0, loader.__len__())

    def test_example_structure(self):
        loader = RelevantSentencesLoader(self.json_path, nltk.sent_tokenize, lambda sent: sent, balance=False)
        # There should be at least one example
        self.assertGreater(loader.__len__(), 0, 'There should be at least one example.')
        for i in range(loader.__len__()):
            example = loader.get_example(i)
            # Test the structure of the example (this should be tested at least once by one of the first assumptions)
            self.assertTrue('sentence' in example, 'The example should contain a "sentence" field.')
            self.assertTrue('features' in example, 'The example should contain a "features" field.')
            self.assertTrue('position' in example, 'The example should contain a "position" field.')
            self.assertTrue('is_relevant' in example, 'The example should contain a "is_relevant" field.')
            self.assertTrue(type(example['is_relevant']) == bool, 'The "is_relevant" field should be a boolean.')

    def test_balanced_loader(self):
        loader = RelevantSentencesLoader(self.json_path, nltk.sent_tokenize, lambda sent: sent, balance=True)
        self.assertTrue(loader.__len__() % 2 == 0, 'There should be an even number of examples with balance=True.')
        # There should be at least one example
        self.assertGreater(loader.__len__(), 0, 'There should be at least one example.')
        pos_examples = 0
        neg_examples = 0
        for i in range(loader.__len__()):
            example = loader.get_example(i)
            if example['is_relevant']:
                pos_examples += 1
            else:
                neg_examples += 1
        self.assertEqual(pos_examples, neg_examples, 'With balance=True, the number of positive examples should equal '
                                                     'the number of negative examples.')


class TestEntitySalienceLoader(TestLoaders):

    def setUp(self):
        super(TestEntitySalienceLoader, self).setUp()
        nltk.download('punkt')
        self.model = EntitySalienceLoader
        self.cache_path = os.path.join(self.resources_path, 'cache')

    def tearDown(self):
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)

    def test_nonexisting_path(self):
        with self.assertRaises(IOError):
            EntitySalienceLoader(os.path.join(self.resources_path, 'nonexisting_directory'))

    def test_max_files(self):
        loader = EntitySalienceLoader(self.json_path, word_vocab_size=10, max_files=0)
        self.assertEqual(0, loader.__len__())

    def test_example_structure(self):
        loader = EntitySalienceLoader(self.json_path, word_vocab_size=100, balance=True)
        # There should be at least one example
        self.assertGreater(loader.__len__(), 0, 'There should be at least one example.')
        for i in range(loader.__len__()):
            example = loader.get_example(i)

            # Test the structure of the example (this should be tested at least once by one of the first assumptions)
            self.assertTrue('features' in example, 'The example should contain an "features" field.')
            self.assertTrue('is_salient' in example, 'The example should contain an "is_salient" field.')
            self.assertTrue(type(example['is_salient']) == bool, 'The "is_salient" field should be a boolean.')
            self.assertTrue('postag_ids' in example, 'The example should contain an "postag_ids" field.')
            self.assertTrue('word_ids' in example, 'The example should contain an "word_ids" field.')
            self.assertTrue('postag_ids' in example, 'The example should contain an "postag_ids" field.')
            self.assertTrue('word_ids' in example, 'The example should contain an "word_ids" field.')

    def test_cache(self):
        loader = EntitySalienceLoader(self.json_path, word_vocab_size=100, balance=False, cache_path=self.cache_path,
                                      remove_duplicate_sentences=True)
        # Load an example
        result1 = loader.get_example(0)
        result2 = loader.get_example(0)

        assert result1 == result2
        # There should at least exists one file by the caching mechanism
        assert len(os.listdir(self.cache_path)) > 0
