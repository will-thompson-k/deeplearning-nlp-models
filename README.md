# deeplearning-nlp-models
[![Coverage Status](https://coveralls.io/repos/github/will-thompson-k/deeplearning-nlp-models/badge.svg?branch=master)](https://coveralls.io/github/will-thompson-k/deeplearning-nlp-models?branch=master)
![Travis (.com)](https://img.shields.io/travis/com/will-thompson-k/deeplearning-nlp-models)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/will-thompson-k/deeplearning-nlp-models)
![GitHub](https://img.shields.io/github/license/will-thompson-k/deeplearning-nlp-models)

A small, interpretable codebase containing the re-implementation of a few "deep" NLP papers in PyTorch.

<ins>Current models</ins>: word2vec, cnn, transformer, gpt.

![Meta](media/bert.jpg)

BERT.Reading.

This is a compact *tutorial* for those interested in getting into the weeds the latest nlp model architectures.
Other repos I found too sprawling to follow. ( <ins>Note</ins>: Most of these models are not adequately trained to be used in production. 
).

## Contents

- [Models](#Models)
- [Features](#Features)
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
|  1. |  [Word2Vec](nlpmodels/notebooks/word2vec/README.md)   | 
|  <ins>CNNs</ins>|             | 
|  2. |  [CNN-based Sentence-level Text Classification](nlpmodels/notebooks/cnn/README.md)   | 
|  <ins>Transformers</ins> |                | 
|  3. |  [The O.G. Transformer](nlpmodels/notebooks/transformer/README.md)  | 
|  4. |  [OpenAI's GPT Model](nlpmodels/notebooks/gpt/README.md)  | 

## Features

This repository has the following features:

- [ ] <ins>model overviews</ins>: A brief overview of each model's motivation and design are provided in separate README.md files.
- [ ] <ins>how-to's</ins>: Jupyter notebooks showing how to run the models and some simple analyses of the model results.
- [ ] <ins>model utilities</ins>: Tokenizers, dataset loaders, dictionaries, and all the custom utilities required for each problem.
- [ ] <ins>multiple dataset APIs</ins>: Both *HuggingFaces* and *torchtext* (i.e. Pytorch) datasets are used in examples.


## Roadmap

Future models:

- [ ] GloVe embeddings
- [ ] Char-RNN (Kaparthy)
- [ ] BERT(s)
- [ ] ELMo
- [ ] XLNet 
- [ ] VAE


Future repo features:

- [ ] Leverage PyTorch **gpu** training (use colab to run, link to open in colab).
- [ ] Gradient clipping
- [ ] saving checkpoints/ loading models
- [ ] Tensorboard plots
- [ ] BPE from openai/gpt-2

## Setup

You can install the repo using `pip`:

```python
pip install git+https://github.com/will-thompson-k/deeplearning-nlp-models 
```

## Structure

Here is a breakdown of the repository:

- [ ] `nlpmodels/models`: The model code for each paper.
- [ ] `nlpmodels/utils`: Contains all the auxiliary classes related to building a model, 
including datasets, vocabulary, tokenizers and trainer classes.
- [ ] `nlpmodels/tests`: Light (and by no means comprehensive) coverage.
- [ ] `nlpmodels/notebooks`: Contains the notebooks and write-ups for each model implementation.

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