import logging

import torch

import helper
from ..models import nodes
from ..models.deeplog import DeepLog

helper.setup_log()
log = logging.getLogger(__name__)


def test_model_creation():
    model = DeepLog(8, 2, 3, 2, "cpu")
    assert model.lstm.hidden_size == 2


def test_model_forward():
    model = DeepLog(4, 2, 3, 2, "cpu")
    out = model.forward(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float).view(-1, 2, 4))
    assert out.size() == torch.Size([1, 2])


def test_make_sequence():
    seq = nodes.make_sequences("test1", [1, 2, 3, 4, 5, 6, 7, 8, 9], 8)
    assert seq == [(1, 2, 3, 4, 5, 6, 7, 8, 9)]

    seq = nodes.make_sequences("test2", [1, 2, 3, 4, 5, 6, 7, 8, 9], 9)
    assert seq == [(1, 2, 3, 4, 5, 6, 7, 8, 9, -1)]
