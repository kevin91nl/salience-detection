import json
import os

import numpy as np
from chainer.dataset import DatasetMixin
from tqdm import tqdm

from dataset.validator import is_valid_esd_json
from util.text import compute_sentence_similarity


class RelevantSentencesLoader(DatasetMixin):

    def __init__(self, path, sent_tokenize, balance=False, seed=0):
        """The file loader for the dataset files as described in the README.

        Parameters
        ----------
        path : str
            Path to the folder containing the JSON files.

        sent_tokenize : callable
            A method which takes a document (text) as input and produces a list of sentences found in the document as
            output.

        balance : bool, optional
            Whether to balance the dataset or not (default: False).

        seed : int, optional
            Seed used for shuffling during balancing (only used when balance=True, default: 0).

        Raises
        ------
        ValueError
            When the path is not a valid directory.
        """
        if not os.path.isdir(path):
            raise IOError('The path "%s" is not a directory.' % path)
        files = os.listdir(path)

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
                relevant_indices = self._compute_relevant_indices(text_sentences, abstract_sentences)
                for index in range(len(text_sentences)):
                    example = {
                        'sentence': text_sentences[index],
                        'is_relevant': index in relevant_indices
                    }
                    self.dataset.append(example)

        if balance:
            np.random.seed(seed)
            pos_examples = [example for example in self.dataset if example['is_relevant']]
            neg_examples = [example for example in self.dataset if not example['is_relevant']]
            min_class_size = min(len(pos_examples), len(neg_examples))

            # Make sure that the examples per class are not more than the minimum class size
            pos_examples = np.random.choice(pos_examples, min_class_size)
            neg_examples = np.random.choice(neg_examples, min_class_size)

            self.dataset = []
            self.dataset.extend(pos_examples)
            self.dataset.extend(neg_examples)
            np.random.shuffle(self.dataset)

    @staticmethod
    def _compute_relevant_indices(text_sentences, abstract_sentences):
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
        set
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

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        return self.dataset[i]
