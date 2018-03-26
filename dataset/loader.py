import json
import os

import numpy as np
from chainer.dataset import DatasetMixin
from tqdm import tqdm

from dataset.validator import is_valid_esd_json
from model.relevant_sentence import RSD
from util.text import compute_sentence_similarity


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

    def __init__(self, path: str, sent_tokenize: callable, sent_to_features: callable, rsd_model: RSD = None,
                 rsd_threshold: float = 0.0, rsd_sent_to_features: callable = None, balance: bool = False,
                 seed: int = 0, max_files: int = -1, remove_duplicate_sentences: bool = True):
        """The file loader for the dataset files for the entity salience task.

        Parameters
        ----------
        path : str
            Path to the folder containing the JSON files.

        sent_tokenize : callable
            A method which takes a document (text) as input and produces a list of sentences found in the document as
            output.

        sent_to_features : callable
            A method which takes a sentence as input and produces features (a vector of embedding indices) as output.

        rsd_model : RSD, optional
            A trained RSD model (default: None).

        rsd_threshold : float, optional
            The minimum relevance score required for a sentence (predicted by the RSD model). Sentences with a lower
            predicted relevance score are omitted (default: 0.0, allows all sentences).

        rsd_sent_to_features : callable, optional
            The sentence-to-features method used for the RSD model (default: None, required when rsd_model is not None).

        balance : bool, optional
            Whether to balance the dataset or not (default: False).

        seed : int, optional
            Seed used for shuffling during balancing (only used when balance=True, default: 0).

        max_files : int, optional
            The maximum number of files used, -1 for no maximum (default: -1).

        remove_duplicate_sentences : bool, optional
            When True, duplicate sentences are removed from the input (default: True).

        Raises
        ------
        ValueError
            When the path is not a valid directory.
        ValueError
            When the rsd_model is specified, but the rsd_sent_to_features method is not specified.
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
                text, entities = file_data['text'], file_data['entities']
                is_valid_esd_json(file_data, is_train_document=True)
                text_sentences = sent_tokenize(text)
                if remove_duplicate_sentences:
                    text_sentences = EntitySalienceLoader._remove_duplicate_sentences(text_sentences)

                # Now filter out sentences using the RSD model
                if rsd_model is not None:
                    if rsd_sent_to_features is None:
                        raise ValueError(
                            'The rsd_model is specified, but the rsd_sent_to_features method is not specified.')
                    scores = rsd_model.predict([rsd_sent_to_features(sentence) for sentence in text_sentences]).data
                    scores = scores[:, 1]
                    text_sentences = [sentence for sentence, score in zip(text_sentences, scores) if
                                      score >= rsd_threshold]

                # Skip if no sentence were found
                if len(text_sentences) == 0:
                    continue

                for entity in entities:
                    example = {
                        'entity': entity,
                        'sentences': text_sentences,
                        'features': [sent_to_features(sentence) for sentence in text_sentences],
                        'is_salient': entity['salience']
                    }
                    self.dataset.append(example)

        if balance:
            self.dataset = balance_dataset(self.dataset, 'is_salient', seed)

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

    def __len__(self) -> int:
        return len(self.dataset)

    def get_example(self, i: int) -> dict:
        return self.dataset[i]
