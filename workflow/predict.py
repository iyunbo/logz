import time

import torch

import workflow.config as config
from deeplog.model import DeepLog


def generate(name, sample):
    hdfs = []
    ln = sample + [-1] * (config.WINDOW_SIZE + 1 - len(sample))
    hdfs.append(tuple(ln))
    print('Number of sequences({}): {}'.format(name, len(ln)))
    return hdfs


def run(num_classes, model_path):
    # Hyperparameters
    device = torch.device(config.DEVICE)
    input_size = config.INPUT_SIZE
    num_layers = config.NUM_LAYERS
    hidden_size = config.HIDDEN_SIZE
    window_size = config.WINDOW_SIZE
    num_candidates = config.NUM_CANDIDATES

    model = DeepLog(input_size, hidden_size, num_layers, num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate('normal', config.NORMAL_SAMPLE)
    test_abnormal_loader = generate('abnormal', config.ABNORMAL_SAMPLE)
    true_positive = 0
    false_positive = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    false_positive += 1
                    break
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    true_positive += 1
                    break
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    false_negative = len(test_abnormal_loader) - true_positive
    precision = 100 * true_positive / (true_positive + false_positive)
    recall = 100 * true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            false_positive, false_negative, precision, recall, f1))
    print('Finished Predicting')
