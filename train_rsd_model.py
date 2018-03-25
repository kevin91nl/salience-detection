import argparse
import json
import os

import chainer
import nltk
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions

from dataset.converter import convert_rsd_batch
from dataset.loader import RelevantSentencesLoader
from model.relevant_sentence import RSD
from util.text import word_to_hash

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Relevant Sentences Detection (RSD) model.')
    parser.add_argument('train_path', type=str,
                        help='Path to the folder containing the JSON files used for training.')
    parser.add_argument('test_path', type=str,
                        help='Path to the folder containing the JSON files used for testing.')
    parser.add_argument('out_path', type=str,
                        help='Path used for storing the output (model files, logs and plots).')

    parser.add_argument('--model-file', default='model.npz', type=str,
                        help='The filename of the model.')
    parser.add_argument('--config-file', default='config.json', type=str,
                        help='The filename of the config file.')
    parser.add_argument('--plot-loss-file', default='loss.png', type=str,
                        help='The filename of the file containing the plot for the loss.')
    parser.add_argument('--plot-f1-file', default='f1.png', type=str,
                        help='The filename of the file containing the plot for the F-1 score.')
    parser.add_argument('--plot-precision-file', default='precision.png', type=str,
                        help='The filename of the file containing the plot for the precision.')
    parser.add_argument('--plot-recall-file', default='recall.png', type=str,
                        help='The filename of the file containing the plot for the recall.')

    parser.add_argument('--vocab-size', default=10000, type=int,
                        help='The number of words in the vocabulary.')
    parser.add_argument('--word-size', default=64, type=int,
                        help='The dimensionality of the word embeddings.')
    parser.add_argument('--hidden-size', default=64, type=int,
                        help='The dimensionality of the hidden states.')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='The dropout rate used during training.')

    parser.add_argument('--train-max-files', default=-1, type=int,
                        help='The maximum number of files to read from the train_path directory '
                             '(-1 for no limit).')
    parser.add_argument('--train-batch-size', default=20, type=int,
                        help='The batch size used during training.')
    parser.add_argument('--train-shuffle', default=True, type=bool,
                        help='Whether to shuffle the examples during training.')
    parser.add_argument('--train-repeat', default=False, type=bool,
                        help='Whether to repeat the examples during training (may cause overfitting when True).')
    parser.add_argument('--train-balance', default=True, type=bool,
                        help='Whether to balance the train dataset.')

    parser.add_argument('--test-max-files', default=20, type=int,
                        help='The maximum number of files to read from the test_path directory '
                             '(-1 for no limit).')
    parser.add_argument('--test-batch-size', default=20, type=int,
                        help='The batch size used during testing.')
    parser.add_argument('--test-shuffle', default=True, type=bool,
                        help='Whether to shuffle the examples during testing.')
    parser.add_argument('--test-repeat', default=False, type=bool,
                        help='Whether to repeat the examples during testing.')
    parser.add_argument('--test-balance', default=True, type=bool,
                        help='Whether to balance the test dataset.')

    parser.add_argument('--optimizer', default='adadelta', choices=['adadelta'],
                        help='The optimizer to use.')

    parser.add_argument('--stop-trigger', default='1 epoch',
                        help='The stop trigger (valid units: "iteration" and "epoch").')
    parser.add_argument('--print-trigger', default='1 iteration',
                        help='The print trigger (valid units: "iteration" and "epoch").')
    parser.add_argument('--log-trigger', default='10 iteration',
                        help='The log trigger (valid units: "iteration" and "epoch").')
    parser.add_argument('--store-model-trigger', default='10 iteration',
                        help='The trigger for storing the model (valid units: "iteration" and "epoch").')

    args = parser.parse_args()


    def parse_trigger(trigger):
        return int(trigger.split()[0]), trigger.split()[1]


    def sentence_to_features(sentence):
        return [word_to_hash(word, args.vocab_size) for word in nltk.word_tokenize(sentence)]


    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    config_file = os.path.join(args.out_path, args.config_file)
    with open(config_file, 'w') as output_file:
        json.dump(vars(args), output_file, indent=4)

    train_loader = RelevantSentencesLoader(args.train_path, nltk.sent_tokenize, sentence_to_features,
                                           balance=args.train_balance,
                                           max_files=args.train_max_files)
    train_iter = SerialIterator(train_loader,
                                batch_size=args.train_batch_size,
                                shuffle=args.train_shuffle,
                                repeat=args.train_repeat)

    test_loader = RelevantSentencesLoader(args.test_path, nltk.sent_tokenize, sentence_to_features,
                                          balance=args.test_balance,
                                          max_files=args.test_max_files)
    test_iter = SerialIterator(test_loader,
                               batch_size=args.test_batch_size,
                               shuffle=args.test_shuffle,
                               repeat=args.test_repeat)

    model = RSD(args.vocab_size, args.word_size, args.hidden_size, args.dropout)

    optimizer = None
    if args.optimizer == 'adadelta':
        optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer,
                                       converter=convert_rsd_batch)
    trainer = training.Trainer(updater, parse_trigger(args.stop_trigger),
                               out=args.out_path)

    trainer.extend(extensions.Evaluator(test_iter, model, converter=convert_rsd_batch),
                   trigger=parse_trigger(args.log_trigger))
    trainer.extend(extensions.LogReport(trigger=parse_trigger(args.log_trigger)))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         file_name=args.plot_loss_file,
                                         trigger=parse_trigger(args.log_trigger)))
    trainer.extend(
        extensions.PlotReport(['main/f1', 'validation/main/f1'],
                              file_name=args.plot_f1_file,
                              trigger=parse_trigger(args.log_trigger)))
    trainer.extend(extensions.PlotReport(['main/recall', 'validation/main/recall'],
                                         file_name=args.plot_recall_file,
                                         trigger=parse_trigger(args.log_trigger)))
    trainer.extend(
        extensions.PlotReport(['main/precision', 'validation/main/precision'],
                              file_name=args.plot_precision_file,
                              trigger=parse_trigger(args.log_trigger)))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/f1', 'main/precision', 'main/recall']),
        trigger=parse_trigger(args.print_trigger))
    trainer.extend(extensions.snapshot_object(model, args.model_file),
                   trigger=parse_trigger(args.store_model_trigger))

    trainer.run()
