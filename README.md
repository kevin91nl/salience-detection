# Entity Salience Detection

![Python 2.7](https://img.shields.io/badge/python-3.6-blue.svg) [![CircleCI](https://circleci.com/gh/kevin91nl/salience_detection/tree/master.svg?style=shield)](https://circleci.com/gh/kevin91nl/salience_detection/tree/master)

The goal of this project is to detect salient entities in text documents.

## Input data

The input files are stored in folders and have a JSON structure. Each JSON file contains one document. It should have a `text` field containing the document text. The documents used during training and testing should have an `entities` field which is a list consisting of entities. An `entity` is a JSON object and should have an `entity` field and a `salience` field. The `entity` field is the most representative mention of the entity in the document. The `salience` field is `true` when the given entity is salient in the document and `false` otherwise.

### Example JSON file

The following JSON file could be used as a training or testing document:

```json
{
  "text": "This text is about dogs. Dogs are animals. Cats are also animals.",
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