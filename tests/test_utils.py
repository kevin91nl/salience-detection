import unittest

from utils import compute_jaccard_index


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


if __name__ == '__main__':
    unittest.main()
