"""step: parsing the log"""
import logging
import os.path as path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from . import drain

log = logging.getLogger(__name__)


def parse_log(input_dir, log_format, log_file, result_dir, similarity_threshold, depth, regex):
    """parsing log
    :param input_dir:
    :param log_format: the log format expression
    :param log_file: the target log file to analyse
    :param result_dir: the output dir for storing the result
    :param similarity_threshold: Similarity threshold
    :param depth: Depth of all leaf nodes
    :param regex: regular expression list for optional pre-processing
    """
    # parsing log into structured CSV
    parser = drain.LogParser(log_format, indir=input_dir, outdir=result_dir, depth=depth,
                             similarity_threshold=similarity_threshold, content_regex=regex)
    parser.parse(log_file)

    return path.join(result_dir, log_file + drain.SUFFIX_PARSED_LOG)


def generate_dataset(series, window_size):
    inputs = []
    outputs = []
    sequence = series.tolist()
    log.info('sequence size: {}'.format(len(sequence)))
    for i in range(len(sequence) - window_size):
        inputs.append(sequence[i:i + window_size])
        outputs.append(sequence[i + window_size])

    dataset = TensorDataset(torch.tensor(
        inputs, dtype=torch.float), torch.tensor(outputs))

    log.info('Number of sequences({}): {}'.format('Event', len(inputs)))
    return dataset


def make_dataloader(csv_file, event_key, sequence_key, window_size, batch_size):
    # read CSV
    df = pd.read_csv(csv_file)

    num_classes = len(df[event_key].unique())
    log.info(f'Total event count: {num_classes}')

    le = LabelEncoder()

    # generate dataset
    df[sequence_key] = le.fit_transform(df[event_key])

    dataset = generate_dataset(df[sequence_key], window_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dataloader, num_classes
