import logging
import os.path as path
import pathlib

import torch
from torch.utils.data import TensorDataset, DataLoader

import helper
from ..models import nodes
from ..models.deeplog import DeepLog

helper.setup_log()
log = logging.getLogger(__name__)
pwd = pathlib.Path(__file__).parent.absolute()
num_epochs = 3


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


def test_training():
    dataset = TensorDataset(torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8],
         [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=torch.float), torch.tensor([0, 1]))
    dl = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    model_file = nodes.train(dataloader=dl, model_dir=path.join(pwd, "data"), num_classes=2, window_size=8,
                             batch_size=1, num_epochs=num_epochs, input_size=1, hidden_size=8, num_layers=2,
                             device="cpu")
    assert model_file.endswith(f"Adam_batch_size=1_epoch={num_epochs}.pt")


def test_prediction():
    nodes.predict(num_classes=2, model_path=path.join(pwd, "data", f"Adam_batch_size=1_epoch={num_epochs}.pt"),
                  normal_sample=[1, 2, 3, 4, 5, 6, 7, 8], abnormal_sample=[2, 2, 2, 2, 2, 2, 2, 2], window_size=3,
                  input_size=1, num_layers=2, num_candidates=2, cpu_or_gpu="cpu", hidden_size=8)
