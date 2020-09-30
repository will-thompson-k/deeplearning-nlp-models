# deeplearning-nlp-models


This repository contains the re-implementation of a handful of "deep" NLP papers in PyTorch.  

Each model implementation is intended to be compact and interpretable and is accompanied by an overview of the paper's 
details and a self-contained Jupyter notebook.

## Contents

- [Models](#Models)
- [Setup](#Setup)
- [Structure](#Structure)
- [Roadmap](#Roadmap)
- [Requirements](#Requirements)
- [Citation](#Citation)
- [License](#License)

## Models

### Embeddings
- [ ] [Word2Vec embeddings (skip-gram)](notebooks/word2vec/README.md)

## Setup

You can install the repo using `pip`:

```python
python -m pip install git+https://github.com/will-thompson-k/deeplearning-nlp-models 
```

## Structure

Here is a breakdown of the repository:

- [ ] `nlpmodels/models`: The model code for each paper.
- [ ] `nlpmodels/utils`: Contains all the auxiliary classes related to building a model, 
including datasets, vocabulary, tokenizers and trainer classes.
- [ ] `nlpmodels/tests`: Coverage.
- [ ] `notebooks/`: Contains the notebooks and write-ups for each model implementation.


## Roadmap

Here are some models I want to implement in the near future:

- [ ] GloVe embeddings
- [ ] Attention / the Transformer
- [ ] BERT (maybe all the BERTs)
- [ ] ELMo
- [ ] GPT-2
- [ ] GPT-3
- [ ] XLNet


## Requirements

Python 3.6+

Here are the package requirements (found in requirements.txt)

datasets==1.0.2  
torch==1.5.1  
pytest==5.4.3  
tqdm==4.45.0  
numpy==1.19.0  


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