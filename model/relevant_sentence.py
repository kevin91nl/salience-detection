import chainer
import chainer.functions as functions
import chainer.links as links
import numpy as np


class RSD(chainer.Chain):

    def __init__(self, vocab_size: int, word_size: int, hidden_size: int, dropout: float = 0.5, **kwargs):
        """Initialize the Relevant Sentences Detector model.

        Parameters
        ----------
        vocab_size : int
            The number of items in the vocabulary of the input.
        word_size : int
            The number of dimensions used to represent the vocabulary.
        hidden_size : int
            The number of dimensions used to represent the hidden state of the LSTM.
        dropout : float, optional
            The dropout used for the LSTM (default: 0.5).
        kwargs
            Optional keyword arguments passed to the super class.
        """
        super().__init__(**kwargs)
        with self.init_scope():
            self.embed = links.EmbedID(vocab_size, word_size)
            self.rnn = links.NStepBiLSTM(1, word_size, hidden_size, dropout)
            self.l_out = links.Linear(2)

    def __call__(self, xs: list, ts: np.ndarray):
        """Compute the loss given an input and an expected output.

        Parameters
        ----------
        xs : list
            A list of sequences where each sequence is a list consisting of integers such that each element is a
            pointer to the vocabulary.
        ts : np.ndarray
            An array with the same length of xs such that ts[i] corresponds to xs[i] and ts[i] is 1 if xs[i] is a
            relevant sentence and ts[i] is 0 if xs[i] is not a relevant sentence. For index i it holds that:
            0 <= i < batch_size. The array should have data type integer (dtype='i').

        Returns
        -------
        The softmax cross entropy loss.
        """
        # Compute the loss
        out = self._compute_logits(xs)
        loss = functions.softmax_cross_entropy(out, ts)

        # Compute precision, recall and F1
        out_softmax = functions.softmax(out)
        precision = functions.precision(out_softmax, ts, 2)[0]
        recall = functions.recall(out_softmax, ts, 2)[0]
        f1 = functions.f1_score(out_softmax, ts, 2)[0]

        # Report the scores
        chainer.report({'precision': precision[0]}, self)
        chainer.report({'recall': recall[0]}, self)
        chainer.report({'f1': f1[0]}, self)
        chainer.report({'loss': loss.data}, self)

        return loss

    def _compute_logits(self, xs: list):
        """Computes logits given a list of sentences.

        Parameters
        ----------
        xs : list
            List of sentences (see __call__() for more information).

        Returns
        -------
        A batch_size x 2 matrix A where A[i, :] are the logits for sentence xs[i]. The index i is the batch number
        (0 <= i < batch_size).
        """
        # Compute the lengths for each sequence (used for splitting afterwards)
        xs_lens = [len(seq) for seq in xs]
        # Now accumulate all the lengths
        xs_lens_sum = self.xp.cumsum(xs_lens)
        # Compute a flat list of all the words used to look up the embeddings quickly
        xs_flat = self.xp.asarray([x for seq in xs for x in seq], dtype='i')
        # Look up the embeddings
        exs_flat = self.embed(xs_flat)
        # Now split the embeddings such that it is in the original sequence structure
        exs = functions.split_axis(exs_flat, xs_lens_sum[:-1], 0)
        # Compute the states of the LSTM over the sentences
        hx, cs, ys = self.rnn(None, None, exs)
        # Find the final states for each sentence and concatenate it in a batch_size x hidden_size matrix
        final = functions.concat([functions.expand_dims(y[-1], 0) for y in ys], axis=0)
        # Apply a linear transformation such that it forms a batch_size x 2 matrix
        out = self.l_out(final)
        return out

    def predict(self, xs: list):
        """Computes relevance probabilities for each sentence.

        Parameters
        ----------
        xs : list
            List of sentences (see __call__() for more information).

        Returns
        -------
        A batch_size x 2 matrix A where A[i, 0] + A[i, 1] = 1.0 and A[i, 1] represents the probability that sentence
        xs[i] is relevant. The index i is the batch number (0 <= i < batch_size).
        """
        return functions.softmax(self._compute_logits(xs))
