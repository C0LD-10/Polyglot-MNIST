"""
Unit tests for CNN and LeNet-5 models.
Run: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from models.cnn_model import CNNModel
from models.lenet_model import LeNet5Spanish


@pytest.fixture
def cnn():
    return CNNModel(num_classes=80)


@pytest.fixture
def lenet():
    return LeNet5Spanish(num_classes=80)


def test_cnn_output_shape(cnn):
    x = torch.randn(4, 1, 28, 28)
    out = cnn(x)
    assert out.shape == (4, 80), f"Expected (4, 80), got {out.shape}"


def test_lenet_output_shape(lenet):
    x = torch.randn(4, 1, 28, 28)
    out = lenet(x)
    assert out.shape == (4, 80), f"Expected (4, 80), got {out.shape}"


def test_cnn_batch_size_1(cnn):
    x = torch.randn(1, 1, 28, 28)
    out = cnn(x)
    assert out.shape == (1, 80)


def test_lenet_batch_size_1(lenet):
    x = torch.randn(1, 1, 28, 28)
    out = lenet(x)
    assert out.shape == (1, 80)


def test_cnn_parameters(cnn):
    n = cnn.count_parameters()
    assert n > 0, "Model should have parameters"
    assert n < 5_000_000, "Model seems too large"


def test_lenet_parameters(lenet):
    n = lenet.count_parameters()
    assert n > 0
    assert n < 1_000_000


def test_cnn_predict(cnn):
    x = torch.randn(8, 1, 28, 28)
    preds = cnn.predict(x)
    assert preds.shape == (8,)
    assert preds.min() >= 0
    assert preds.max() < 80


def test_cnn_predict_proba(cnn):
    x = torch.randn(4, 1, 28, 28)
    proba = cnn.predict_proba(x)
    assert proba.shape == (4, 80)
    assert torch.allclose(proba.sum(dim=1), torch.ones(4), atol=1e-5)


def test_cnn_train_mode(cnn):
    cnn.train()
    x = torch.randn(2, 1, 28, 28)
    out = cnn(x)
    assert out.shape == (2, 80)


def test_cnn_eval_mode(cnn):
    cnn.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, 28, 28)
        out = cnn(x)
    assert out.shape == (2, 80)


def test_cnn_repr(cnn):
    r = repr(cnn)
    assert "CNNModel" in r
    assert "80" in r


def test_different_num_classes():
    for n in [10, 52, 80]:
        model = CNNModel(num_classes=n)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, n)
