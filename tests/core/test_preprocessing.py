import pytest
import tensorflow as tf
import torch
import numpy as np

from adversarial_lab.core.preprocessing import *
from adversarial_lab.core.preprocessing import PreprocessingFromFunction


@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_preprocessing_instantiation(framework):
    class DummyPreprocessing(Preprocessing):
        def preprocess(self, sample):
            return sample

    preprocessing = DummyPreprocessing()
    assert hasattr(preprocessing, "preprocess")


@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_no_preprocessing(framework):
    preprocessing = NoPreprocessing()

    if framework == "tf":
        sample = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        output = preprocessing.preprocess(sample)
        assert tf.reduce_all(tf.equal(sample, output))

    else:
        sample = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        output = preprocessing.preprocess(sample)
        assert torch.equal(sample, output)


@pytest.mark.parametrize("function_type", ["valid", "missing_sample"])
def test_preprocessing_from_function(function_type):
    if function_type == "missing_sample":
        def invalid_function(*args, **kwargs):
            return args

        with pytest.raises(TypeError, match="must have parameter: 'sample'"):
            PreprocessingFromFunction.create(invalid_function)

    if function_type == "valid":
        def valid_function(sample, *args, **kwargs):
            return sample * 2

        preprocessing = PreprocessingFromFunction.create(valid_function)

        sample = tf.constant([1.0, -1.0, 0.5], dtype=tf.float32)
        result = preprocessing.preprocess(sample)

        assert np.array_equal(result.numpy(), [2.0, -2.0, 1.0])
