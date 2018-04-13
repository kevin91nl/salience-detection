import json
import os

import nltk
import numpy as np
import unidecode
from chainer.dataset import DatasetMixin
from fnvhash import fnv1a_32
from tqdm import tqdm

from dataset.validator import is_valid_esd_json
from util.text import compute_sentence_similarity, compute_levenshtein_score


def balance_dataset(dataset: list, key: str, seed: int = 0) -> list:
    """Balance a dataset.

    Parameters
    ----------
    dataset : list
        List of examples to balance (each example must have a field which is specified in the "key" argument).

    key : str
        The field (binary field consisting of True and False values) to balance for.

    seed : int, optional
        The seed to use for shuffling (default: 0).

    Returns
    -------
    The balanced dataset.
    """
    np.random.seed(seed)
    pos_examples = [example for example in dataset if example[key]]
    neg_examples = [example for example in dataset if not example[key]]
    min_class_size = min(len(pos_examples), len(neg_examples))

    # Make sure that the examples per class are not more than the minimum class size
    pos_examples = np.random.choice(pos_examples, min_class_size)
    neg_examples = np.random.choice(neg_examples, min_class_size)

    dataset = []
    dataset.extend(pos_examples)
    dataset.extend(neg_examples)
    np.random.shuffle(dataset)

    return dataset


class RelevantSentencesLoader(DatasetMixin):

    def __init__(self, path: str, sent_tokenize: callable, sent_to_features: callable, balance: bool = False,
                 seed: int = 0, max_files: int = -1):
        """The file loader for the dataset files as described in the README.

        Parameters
        ----------
        path : str
            Path to the folder containing the JSON files.

        sent_tokenize : callable
            A method which takes a document (text) as input and produces a list of sentences found in the document as
            output.

        sent_to_features : callable
            A method which takes a sentence as input and produces features (a vector of embedding indices) as output.

        balance : bool, optional
            Whether to balance the dataset or not (default: False).

        seed : int, optional
            Seed used for shuffling during balancing (only used when balance=True, default: 0).

        max_files : int, optional
            The maximum number of files used, -1 for no maximum (default: -1).

        Raises
        ------
        ValueError
            When the path is not a valid directory.
        """
        if not os.path.isdir(path):
            raise IOError('The path "%s" is not a directory.' % path)
        files = os.listdir(path)
        if max_files != -1:
            files = files[:max_files]

        # Create a list containing the dataset
        self.dataset = []

        progressbar = tqdm(files)
        for file in progressbar:
            progressbar.set_description(file)
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as input_file:
                file_data = json.load(input_file)
                text, abstract = file_data['text'], file_data['abstract']
                is_valid_esd_json(file_data, is_train_document=True)
                text_sentences = sent_tokenize(text)
                abstract_sentences = sent_tokenize(abstract)
                try:
                    relevant_indices = self._compute_relevant_indices(text_sentences, abstract_sentences)
                except ValueError:
                    continue
                for index in range(len(text_sentences)):
                    example = {
                        'sentence': text_sentences[index],
                        'features': sent_to_features(text_sentences[index]),
                        'position': index / float(len(text_sentences)),
                        'is_relevant': index in relevant_indices
                    }
                    self.dataset.append(example)

        if balance:
            self.dataset = balance_dataset(self.dataset, 'is_relevant', seed)

    @staticmethod
    def _compute_relevant_indices(text_sentences: list, abstract_sentences: list) -> set:
        """Computes the indices of the sentences in the text which are relevant (i.e. the sentences that are described
        in the abstract).

        Parameters
        ----------
        text_sentences : list
            A list of sentences of the text.
        abstract_sentences : list
            A list of sentences of the abstract.

        Returns
        -------
        A set of indices such that for index i text_sentences[i] is relevant.
        """
        relevant_indices = set()
        for abstract_sentence in abstract_sentences:
            scores = []
            for text_sentence in text_sentences:
                score = compute_sentence_similarity(abstract_sentence, text_sentence)
                scores.append(score)
            if np.max(scores) > 0.:
                relevant_indices.add(np.argmax(scores))
            for index, score in enumerate(scores):
                if score > 0.6:
                    relevant_indices.add(index)
        return relevant_indices

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, i: int) -> dict:
        return self.dataset[i]


class EntitySalienceLoader(DatasetMixin):

    def __init__(self, path: str, balance: bool = False, postag_vocab_size: int = 100, word_vocab_size: int = 100000,
                 seed: int = 0, max_files: int = -1, remove_duplicate_sentences: bool = True,
                 max_ent_words: int = 5, cache_path: str = None):
        """The file loader for the dataset files for the entity salience task.

        Parameters
        ----------
        path : str
            Path to the folder containing the JSON files.

        balance : bool, optional
            Whether to balance the dataset or not (default: False).

        seed : int, optional
            Seed used for shuffling during balancing (only used when balance=True, default: 0).

        max_files : int, optional
            The maximum number of files used, -1 for no maximum (default: -1).

        remove_duplicate_sentences : bool, optional
            When True, duplicate sentences are removed from the input (default: True).

        max_ent_words : int, optional
            The maximum number of words used in entities (default: 5).

        cache_path : str, optional
            When specified, the preprocessing of the data is stored in this folder.

        Raises
        ------
        ValueError
            When the path is not a valid directory.
        """
        if not os.path.isdir(path):
            raise IOError('The path "%s" is not a directory.' % path)
        files = os.listdir(path)
        if max_files != -1:
            files = files[:max_files]

        # Create a list containing the dataset
        self.dataset = []

        # Store the arguments
        self.remove_duplicate_sentences = remove_duplicate_sentences

        # Check the vocab arguments
        self.postag_vocab_size = postag_vocab_size
        self.word_vocab_size = word_vocab_size

        self.max_ent_words = max_ent_words

        progressbar = tqdm(files)
        for file in progressbar:
            progressbar.set_description(file)
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as input_file:
                file_data = json.load(input_file)
                text, entities = file_data['text'], file_data['entities']

                is_valid_esd_json(file_data, is_train_document=True)

                for query_entity in entities:
                    example = {
                        '_path': file_path,
                        '_doc_id': str(file_data['doc_id']) if 'doc_id' in file_data else None,
                        '_ent_id': str(query_entity['entity_id']) if 'entity_id' in query_entity else None,
                        'query_entity': query_entity['entity'],
                        'context_entities': [ent['entity'] for ent in entities if
                                             ent['entity'] != query_entity['entity']],
                        'text': text,
                        'is_salient': query_entity['salience']
                    }
                    self.dataset.append(example)

        if balance:
            self.dataset = balance_dataset(self.dataset, 'is_salient', seed)

        self.cache_path = cache_path
        if self.cache_path is not None:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)

    @staticmethod
    def _remove_duplicate_sentences(sentences: list) -> list:
        """Remove duplicate sentences.

        Parameters
        ----------
        sentences : list
            List of sentences (strings).

        Returns
        -------
        List of sentences such that the first occurrence of each unique sentence is ordered the same way as in the input
        data.
        """
        unique_sentences = []
        for sentence in sentences:
            if sentence.lower().strip() not in [sentence.lower().strip() for sentence in unique_sentences]:
                unique_sentences.append(sentence)
        return unique_sentences

    def _word_to_feature(self, word: str) -> int:
        """Converts a word to an identifier.

        Parameters
        ----------
        word : str
            The word to convert.

        Returns
        -------
        An identifier (int) which is either computed by a specified vocabulary or by using a hashing vectorizer.
        """
        return fnv1a_32(unidecode.unidecode(word.lower()).encode('utf-8')) % self.word_vocab_size

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, i: int) -> dict:
        example = self.dataset[i]

        doc_cache_path, ent_cache_path = None, None
        doc_example, ent_example = None, None
        if self.cache_path is not None:
            doc_hash = fnv1a_32(example['text'].encode('utf-8')) % 1000000000
            doc_cache_path = os.path.join(self.cache_path, '%s.json' % doc_hash)
            if os.path.exists(doc_cache_path):
                with open(doc_cache_path, 'r') as input_file:
                    doc_example = json.load(input_file)

            ent_hash = fnv1a_32(example['query_entity'].encode('utf-8')) % 1000000000
            ent_cache_path = os.path.join(self.cache_path, '%s_%s.json' % (doc_hash, ent_hash))
            if os.path.exists(ent_cache_path):
                with open(ent_cache_path, 'r') as input_file:
                    ent_example = json.load(input_file)

        if doc_example is None:
            sents = nltk.sent_tokenize(example['text'])
            if self.remove_duplicate_sentences:
                sents = self._remove_duplicate_sentences(sents)
            annotated_words = [nltk.pos_tag(nltk.word_tokenize(sent)) + [('EOS', 'EOS')] for sent in sents]
            postags = [word_data[1] for sent in annotated_words for word_data in sent]
            words = [word_data[0] for sent in annotated_words for word_data in sent]
            lowered_words = [word.lower() for word in words]

            postag_labels = ['UNK', 'EOS', '(', ')', ',', '--', '.', ':', '$', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
                             'JJR', 'JJS',
                             'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
                             'RP',
                             'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',
                             '``']
            postag_labels = {
                item: i
                for i, item in enumerate(postag_labels)
            }

            has_uppercase_features = [[1. if word.lower() != word else 0.] for word in words]
            has_number_features = [[1. * any(c.isdigit() for c in word)] for word in words]
            has_dots_features = [[1. if '.' in word else 0.] for word in words]

            features = [has_uppercase_features, has_number_features, has_dots_features]

            postag_ids = [postag_labels.get(postag, postag_labels.get('UNK')) % self.postag_vocab_size for postag in
                          postags]
            word_ids = [self._word_to_feature(word) % self.word_vocab_size for word in
                        lowered_words]

            doc_example = {
                'features': features,
                'postag_ids': postag_ids,
                'word_ids': word_ids,
                'words': words,
                'postags': postags
            }
        else:
            lowered_words = [word.lower() for word in doc_example['words']]

        if ent_example is None:
            context_entities = example['context_entities']
            if len(context_entities) == 0:
                context_entities = ['']
            context_entities_words = [nltk.word_tokenize(ent)[:self.max_ent_words] for ent in context_entities]
            context_entities_words = [entity_words + ['' for _ in range(self.max_ent_words - len(entity_words))] for
                                      entity_words
                                      in context_entities_words]

            query_entity_words = nltk.word_tokenize(example['query_entity'])[:self.max_ent_words]
            query_entity_words += ['' for _ in range(self.max_ent_words - len(query_entity_words))]

            query_entity_features = [[compute_levenshtein_score(word, ent_word) for ent_word in query_entity_words] for
                                     word
                                     in lowered_words]
            context_entities_features = [[max(
                [compute_levenshtein_score(word, ent_words[index]) for ent_words in context_entities_words]) for index
                in
                range(self.max_ent_words)] for word in lowered_words]

            ent_example = {
                'is_salient': example['is_salient'],
                'context_entities_words': context_entities_words,
                'query_entity_words': query_entity_words,
                'query_entity_features': query_entity_features,
                'context_entities_features': context_entities_features
            }

        if doc_cache_path is not None and not os.path.exists(doc_cache_path):
            with open(doc_cache_path, 'w') as output_file:
                json.dump(doc_example, output_file)

        if ent_cache_path is not None and not os.path.exists(ent_cache_path):
            with open(ent_cache_path, 'w') as output_file:
                json.dump(ent_example, output_file)

        return {**doc_example, **ent_example}
