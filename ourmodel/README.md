### Getting Started
To run the classaifier
```
python run.py [-h] [-a ANNOTATIONS_FILE] [-e EMBEDDINGS_FILE] [-m MAX_INPUT_SEQUENCE_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  -a ANNOTATIONS_FILE   The path to the annotations file. This file contains comments and the corresponding
                        annotations.
  -e EMBEDDINGS_FILE    The path to the embeddings. We use fasttext to train the word embeddings.
  -m MAX_INPUT_SEQUENCE_LENGTH
                        Specifies the maximum tokens from the comment to be consider for classification.
```