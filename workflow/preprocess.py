import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

import workflow.config as config


def generate_dataset(series):
    inputs = []
    outputs = []
    sequence = series.tolist()
    print('sequence size: {}'.format(len(sequence)))
    for i in range(len(sequence) - config.WINDOW_SIZE):
        inputs.append(sequence[i:i + config.WINDOW_SIZE])
        outputs.append(sequence[i + config.WINDOW_SIZE])

    dataset = TensorDataset(torch.tensor(
        inputs, dtype=torch.float), torch.tensor(outputs))

    print('Number of sequences({}): {}'.format('Event', len(inputs)))
    return dataset


def run(csv_file):
    # read CSV
    df = pd.read_csv(csv_file)

    num_classes = len(df[config.EVENT_KEY].unique())
    print('Total event count:', num_classes)

    le = LabelEncoder()

    # generate dataset
    df[config.SEQUENCE_KEY] = le.fit_transform(df[config.EVENT_KEY])

    dataset = generate_dataset(df[config.SEQUENCE_KEY])

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, pin_memory=True)

    return dataloader, num_classes
