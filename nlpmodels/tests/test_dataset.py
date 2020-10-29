from nlpmodels.utils import dataset
import unittest


class TestAbstractDataset(unittest.TestCase):
    def test_cannot_instantiate_abstract_class(self):
        with self.assertRaises(TypeError):
            dataset.AbstractNLPDataset()
