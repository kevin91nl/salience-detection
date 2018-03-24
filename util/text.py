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
        # 997 is a large prime number (larger than the value of the ord() method)
        hash_sum += (997 * (i + 1) * ord(char)) % vocab_size
        hash_sum = hash_sum % vocab_size
    return hash_sum
