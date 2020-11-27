from setuptools import setup

setup(
    name='deeplearning-nlp-models',
    version='1.0',
    packages=['nlpmodels', 'nlpmodels.utils', 'nlpmodels.models', 'nlpmodels.models.transformer_blocks'],
    setup_requires=['pytest-runner','torchtext==0.7.0','torch==1.6.0','numpy==1.19.1','datasets==1.1.2','tqdm==4.49.0'],
    url='https://github.com/will-thompson-k/deeplearning-nlp-models',
    license='MIT',
    author='Will Thompson',
    author_email='',
    description='A repository containing the re-implementation of a handful of "deep" NLP models in PyTorch',
    test_suite="tests",
    tests_require=['pytest'],
)
