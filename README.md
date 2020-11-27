# deeplearning-nlp-models
[![Coverage Status](https://coveralls.io/repos/github/will-thompson-k/deeplearning-nlp-models/badge.png?branch=master)](https://coveralls.io/github/will-thompson-k/deeplearning-nlp-models?branch=master)
![Travis (.com)](https://img.shields.io/travis/com/will-thompson-k/deeplearning-nlp-models)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/will-thompson-k/deeplearning-nlp-models)
![GitHub](https://img.shields.io/github/license/will-thompson-k/deeplearning-nlp-models)

A small, interpretable codebase containing the re-implementation of a few "deep" NLP models in PyTorch.

<ins>Current models</ins>: word2vec, CNNs, transformer, gpt.

![Meta](media/bert.jpg)

BERT: Reading. Comprehending. 

This is a compact primer for those interested in starting to get into the weeds of deep NLP model architectures.
Some other repos I found a bit too sprawling to follow.
( <ins>Note</ins>: These models are toy versions of each model. They are not adequately trained to be used in production. 
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

|    Model Class               |           Model               |   Year                        | 
| :-------------------- | :--------------------  | :--------------------  | 
|  <ins>Embeddings</ins>|             |              | 
|  1. |  [Word2Vec Embeddings (Self-Supervised Learning)](notebooks/word2vec/README.md)   |       2013       | 
|  <ins>CNNs</ins>|             |              | 
|  2. |  [CNN-based Text Classification (Binary Classification)](notebooks/cnn/README.md)   |    2014          | 
|  <ins>Transformers</ins> |                |              | 
|  3. |  [The O.G. Transformer (Machine Translation)](notebooks/transformer/README.md)  |      2017        | 
|  4. |  [OpenAI's GPT Model (Language Model)](notebooks/gpt/README.md)  |   2018, 2019, 2020           | 

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

Future repo features:

- [ ] Leverage PyTorch **gpu** training (use colab to run, link to open in colab).
- [ ] Gradient clipping
- [ ] Tensorboard plots
- [ ] Val set demonstrations
- [ ] Saving checkpoints/ loading models
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
- [ ] `tests`: Light (and by no means comprehensive) coverage.
- [ ] `notebooks`: Contains the notebooks and write-ups for each model implementation.
- [ ] `run_tests.sh`: If you want to run the tests yourself (you can also use `setup.py test`).
*Warning*: Some of the regression tests cause the whole suite to take a few mins to run.
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
