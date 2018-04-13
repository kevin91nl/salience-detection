import re

import nltk
import unidecode

from util.metric import compute_jaccard_index


def compute_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """Compute the similarity using the Jaccard index on the BOW model of both sentences.

    Parameters
    ----------
    sentence1 : str
        The first sentence.
    sentence2 : str
        The second sentence.

    Returns
    -------
    The Jaccard index computed on the BOW of both sentences.
    """
    bow1 = get_bow(sentence1)
    bow2 = get_bow(sentence2)
    return compute_jaccard_index(bow1, bow2)


def get_bow(sentence: str) -> set:
    """Compute the Bag-of-Words (BOW) set of a sentence.

    Parameters
    ----------
    sentence : str
        Sentence to compute the Bag-of-Words representation for.

    Returns
    -------
    The set of unique cleaned words (cleaned by the clean_word method) found in the sentence with non-zero length.
    """
    words = nltk.word_tokenize(sentence)
    return {clean_word(word) for word in words if len(clean_word(word)) > 0}


def clean_word(word: str) -> str:
    """Clean a word: remove any non-alphabetic character of the lower-cased version of the word and remove any accents.

    Parameters
    ----------
    word : str
        Word to clean.

    Returns
    -------
    str
        Cleaned word.
    """
    return re.sub(r'[^a-z]+', '', unidecode.unidecode(word).lower())


def word_to_hash(word: str, vocab_size: int) -> int:
    """Compute a hash for a word.

    Parameters
    ----------
    word : str
        Word to compute the hash for.

    vocab_size : int
        The maximum number of words in the vocab.

    Returns
    -------
    int
        The hash such that 0 <= hash < vocab_size.
    """
    cleaned_word = clean_word(word)
    hash_sum = 0
    for i, char in enumerate(list(cleaned_word)):
        # A large prime number is used for better pseudo-randomness approximation
        hash_sum += (i * ord(char) * 104729) % vocab_size
        hash_sum = hash_sum % vocab_size
    return hash_sum


def levenshtein(seq1, seq2):
    """Compute the edit distance between two words seq1 and seq2.

    Parameters
    ----------
    seq1 : str
        The first word.
    seq2 : str
        The second word.

    Returns
    -------
    int
        The edit distance.
    """
    if len(seq1) < len(seq2):
        return levenshtein(seq2, seq1)

    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[
                             j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_levenshtein_score(w1, w2):
    """Compute a score based on the edit distance for two words w1 and w2.

    Parameters
    ----------
    w1 : str
        The first word.
    w2 : str
        The second word.

    Returns
    -------
    float
        0.0 if either w1 is empty or w2 is empty.
        1.0 - levenshtein(w1, w2) / max_possible such that identical words have score 1.0 and completely dissimilar
        words have score 0.0.
    """
    if min(len(w1), len(w2)) == 0:
        return 0.
    max_levenshtein = max(len(w1), len(w2))
    return 1. - levenshtein(unidecode.unidecode(w1.lower()), unidecode.unidecode(w2.lower())) / float(max_levenshtein)
