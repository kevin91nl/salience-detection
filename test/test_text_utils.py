import unittest

import nltk

from util.text import clean_word, get_bow, compute_sentence_similarity


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


if __name__ == '__main__':
    unittest.main()
