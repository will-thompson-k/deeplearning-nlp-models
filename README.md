# deeplearning-nlp-models


This repository contains the re-implementation of a handful of "deep" NLP papers in PyTorch.  

Each model implementation is intended to be compact and interpretable and is accompanied by a brief overview of the paper's 
details as well as a self-contained Jupyter notebook.

## Contents

- [Features](#Features)
- [Models](#Models)
- [Setup](#Setup)
- [Structure](#Structure)
- [Roadmap](#Roadmap)
- [Requirements](#Requirements)
- [Citation](#Citation)
- [License](#License)

## Features

This repository has the following features:

- [ ] **Model Breakdowns**: A brief explanation of the model and its components are provided in separate README.md files.
- [ ] **Jupyter Notebooks**: Model instantiation and running are provided for each model.
- [ ] **Complete Model Utilities**: Tokenizers, datasetloaders, dictionaries, and all the custom utilities required for each problem.
- [ ] **Documented Code**: Classes and methods are described in docstrings and comments.
- [ ] **Multiple Dataset Libraries**: Both *HuggingFaces* and *torchtext* (i.e. Pytorch) datasets are used in examples.

## Models

### Embeddings
- [ ] [Word2Vec::Skip-gram embeddings (Negative Sampling)](nlpmodels/notebooks/word2vec/README.md)

### Sequence-to-Sequence Models

#### Transformers
- [ ] [O.G. Transformer ("Attention is All You Need")](nlpmodels/notebooks/transformer/README.md)


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
- [ ] `nlpmodels/tests`: Coverage.
- [ ] `nlpmodels/notebooks`: Contains the notebooks and write-ups for each model implementation.


## Roadmap

Here are some models I want to implement in the near future:

- [ ] GloVe embeddings
- [ ] TextCNN
- [ ] Encoder-Decoder RNN
- [ ] GPT
- [ ] BERT (maybe all the BERTs)
- [ ] ELMo
- [ ] XLNet
- [ ] T5 (also by Google: Performer attention-model)

## Requirements

You can install the requirements here:

```python
pip install -r requirements.txt 
```

Python 3.6+

Here are the package requirements (found in requirements.txt)

- [ ] numpy==1.19.1
- [ ] tqdm==4.50.2
- [ ] torch==1.6.0
- [ ] datasets==1.0.2
- [ ] torchtext==0.8.0a0+c851c3e


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