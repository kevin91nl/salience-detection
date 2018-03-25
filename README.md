# Salience Detection

![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg) [![CircleCI](https://circleci.com/gh/kevin91nl/salience-detection/tree/master.svg?style=shield)](https://circleci.com/gh/kevin91nl/salience-detection/tree/master) [![codecov](https://codecov.io/gh/kevin91nl/salience-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/kevin91nl/salience-detection)

The goal of this project is to detect salient entities in text documents.

## Input data

The input files are stored in folders and have a JSON structure. Each JSON file contains one document. It should have a `text` field containing the document text. The documents used during training and testing should have an `abstract` field consisting of the abstract of the text and an `entities` field which is a list consisting of entities. An `entity` is a JSON object and should have an `entity` field and a `salience` field. The `entity` field is the most representative mention of the entity in the document. The `salience` field is `true` when the given entity is salient in the document and `false` otherwise.

### Example JSON file

The following JSON file could be used as a training or testing document:

```json
{
  "text": "This text is about dogs. Dogs are animals. Cats are also animals.",
  "abstract": "Text about dogs.",
  "entities": [
    {
      "entity": "dogs",
      "salience": true
    },
    {
      "entity": "cats",
      "salience": false
    }
  ]
}
```

For evaluating a document, only the `text` field is required:

```json
{
  "text": "This text is about dogs. Dogs are animals. Cats are also animals."
}
```

The entities and salience information are determined by the model.

## Relevant Sentence Detection (RSD) model

The first model is the relevant sentence detection (RSD) model. This model computes a relevance score for each sentence and reflects whether the sentence is relevant for the abstract of the text or not. The model is useful for text compression.

### Train the model

In order to train the model, execute the following script:

```
python train_rsd_model.py [train_path] [test_path] [out_path]
```

The `train_path` argument should point to a directory containing input files used for training. The `test_path` argument should point to a directory containing input files used for testing and the `out_path` should be an empty directory in which the model files, log files and plot files should be stored. All model parameters are configurable. The parameters are explained by the help function which can be found by executing the following code:

```
python train_rsd_model.py --help
```