import unittest

import nltk

from util.text import clean_word, get_bow, compute_sentence_similarity, word_to_hash


class TestTextUtils(unittest.TestCase):

    def setUp(self):
        nltk.download('punkt')

    def test_clean_word(self):
        self.assertEqual('hello', clean_word('Héllo!'))
        self.assertEqual('', clean_word('.'))

    def test_get_bow(self):
        self.assertEqual({'hello', 'world'}, get_bow('hello world. hello world!'))
        self.assertEqual({'hello', 'world'}, get_bow('Héllo world!'))

    def test_compute_sentence_similarity(self):
        self.assertEqual(1., compute_sentence_similarity('hello world.', 'Héllo world!'))
        self.assertEqual(0., compute_sentence_similarity('test sentence.', 'Héllo world!'))
        self.assertEqual(1. / (3. + 3. - 1.), compute_sentence_similarity('one shared word.', 'one two three.'))

    def test_word_to_hash(self):
        self.assertEqual(word_to_hash('word', 1000), word_to_hash('Word', 1000))
        # Here, the assumption is that the collision rate is extremely small. In that case, we can assume that
        # hash('word') != hash('different').
        self.assertNotEqual(word_to_hash('word', 100000), word_to_hash('different', 100000))
