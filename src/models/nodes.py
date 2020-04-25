import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.deeplog import DeepLog

log = logging.getLogger(__name__)


def make_sequences(name, sample, window_size):
    sequences = []
    for line in sample:
        ln = line + [-1] * (window_size + 1 - len(line))
        sequences.append(tuple(ln))
    log.info(f'Number of sequences({name}): {len(sequences)}')
    return sequences


def predict(num_classes,
            model_path,
            normal_sample,
            abnormal_sample,
            window_size,
            input_size,
            hidden_size,
            num_layers,
            num_candidates,
            cpu_or_gpu):
    device = torch.device(cpu_or_gpu)
    model = DeepLog(input_size, hidden_size, num_layers, num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    log.info('model_path: {}'.format(model_path))
    test_normal_loader = make_sequences('normal', normal_sample, window_size)
    test_abnormal_loader = make_sequences('abnormal', abnormal_sample, window_size)
    true_positive = 0
    false_positive = 0

    # Test the model
    start_time = time.time()
    false_positive = test_model(device, false_positive, input_size, model, num_candidates, test_normal_loader,
                                window_size)
    true_positive = test_model(device, true_positive, input_size, model, num_candidates, test_abnormal_loader,
                               window_size)
    elapsed_time = time.time() - start_time
    log.info('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    false_negative = len(test_abnormal_loader) - true_positive
    precision = 100 * true_positive / (true_positive + false_positive)
    recall = 100 * true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    log.info(
        f'false positive (FP): {false_positive}, false negative (FN): {false_negative}, Precision: {precision:.2f}%, '
        f'Recall: {recall:.2f}%, F1-measure: {f1:.2f}%')
    log.info('Finished Predicting')


def test_model(device, positive, input_size, model, num_candidates, test_data_loader, window_size):
    with torch.no_grad():
        for line in test_data_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    positive += 1
                    break
    return positive


def train(dataloader, model_dir, num_classes, window_size, batch_size, num_epochs, input_size, hidden_size, num_layers,
          device) -> str:
    model = DeepLog(input_size, hidden_size, num_layers, num_classes, device)
    model_name = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
    writer = SummaryWriter(log_dir=os.path.join(model_dir, model_name))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        log.info('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    log.info('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_file = model_dir + '/' + model_name + '.pt'
    torch.save(model.state_dict(), model_file)
    writer.close()
    log.info('Finished Training')
    return model_file
