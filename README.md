# deeplearning-nlp-models
[![Coverage Status](https://coveralls.io/repos/github/will-thompson-k/deeplearning-nlp-models/badge.svg?branch=master)](https://coveralls.io/github/will-thompson-k/deeplearning-nlp-models?branch=master)
![Travis (.com)](https://img.shields.io/travis/com/will-thompson-k/deeplearning-nlp-models)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/will-thompson-k/deeplearning-nlp-models)
![GitHub](https://img.shields.io/github/license/will-thompson-k/deeplearning-nlp-models)

A small, interpretable codebase containing the re-implementation of a few "deep" NLP models in PyTorch.

<ins>Current models</ins>: word2vec, CNNs, transformer, gpt.

![Meta](media/bert.jpg)

BERT: Reading. Comprehending. 

This is a compact review for those interested in getting into the weeds of dl-nlp model architectures.
Other repos I found too sprawling to follow. This project sprung out of my own self-study. 
( <ins>Note</ins>: These models are not adequately trained to be used in production, nor are they being
benchmarked against a val/test set. 
)

## Contents

- [Models](#Models)
- [Features](#Features)
- [Endgame](#Endgame)
- [Roadmap](#Roadmap)
- [Setup](#Setup)
- [Structure](#Structure)
- [Requirements](#Requirements)
- [Citation](#Citation)
- [License](#License)


## Models

These NLP models are presented chronologically and, as you might expect, build off each other.

|    Model Directory               |                           | 
| :-------------------- | :--------------------  | 
|  <ins>Embeddings</ins>|             | 
|  1. |  [Word2Vec Embeddings (Self-Supervised Learning)](nlpmodels/notebooks/word2vec/README.md)   | 
|  <ins>CNNs</ins>|             | 
|  2. |  [CNN-based Text Classification (Binary Classification)](nlpmodels/notebooks/cnn/README.md)   | 
|  <ins>Transformers</ins> |                | 
|  3. |  [The O.G. Transformer (Machine Translation)](nlpmodels/notebooks/transformer/README.md)  | 
|  4. |  [OpenAI's GPT Model (Language Model)](nlpmodels/notebooks/gpt/README.md)  | 

## Features

This repository has the following features:

- [ ] <ins>model overviews</ins>: A brief overview of each model's motivation and design are provided in separate README.md files.
- [ ] <ins>how-to's</ins>: Jupyter notebooks showing how to run the models and some simple analyses of the model results.
- [ ] <ins>model utilities</ins>: Tokenizers, dataset loaders, dictionaries, and all the custom utilities required for each problem.
- [ ] <ins>multiple dataset APIs</ins>: Both *HuggingFaces* and *torchtext* (i.e. Pytorch) datasets are used in examples.

## Endgame

After reviewing these models, the world's your oyster in terms of other models to explore:

ELMO, XLNET, all the other BERTs, BART, Performer, T5, etc....

## Roadmap

Future models:

- [ ] Char-RNN (Kaparthy)
- [ ] BERT
- [ ] VAE

Future repo features:

- [ ] Leverage PyTorch **gpu** training (use colab to run, link to open in colab).
- [ ] Gradient clipping
- [ ] Val set early stopping
- [ ] Saving checkpoints/ loading models
- [ ] Tensorboard plots
- [ ] BPE (from either openai/gpt-2 or facebook's fairseq library)

## Setup

You can install the repo using `pip`:

```python
pip install git+https://github.com/will-thompson-k/deeplearning-nlp-models 
```

## Structure

Here is a breakdown of the repository:

- [ ] `nlpmodels/models`: The model code for each paper.
- [ ] `nlpmodels/utils`: Contains all the auxiliary classes related to building a model, 
including datasets, vocabulary, tokenizers, samplers and trainer classes.
- [ ] `nlpmodels/tests`: Light (and by no means comprehensive) coverage.
- [ ] `nlpmodels/notebooks`: Contains the notebooks and write-ups for each model implementation.
- [ ] `run_tests.sh`: If you want to run the tests yourself (you can also use `setup.py test`).
- [ ] `run_pylint.sh`: If you really like linting code.

## Requirements

Python 3.6+

Here are the package requirements (found in requirements.txt)

- [ ] numpy==1.19.1
- [ ] tqdm==4.50.2
- [ ] torch==1.6.0
- [ ] datasets==1.0.2
- [ ] torchtext==0.7.0


## Citation

```python 
@misc{deeplearning-nlp-models,
  author = {Thompson, Will},
  url = {https://github.com/will-thompson-k/deeplearning-nlp-models},
  year = {2020}
}
```
## License

MIT