import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import workflow.config as config
from deeplog.model import DeepLog


def run(dataloader, num_classes):
    model = DeepLog(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, num_classes, config.DEVICE)
    log = 'Adam_batch_size={}_epoch={}'.format(str(config.BATCH_SIZE), str(config.NUM_EPOCHS))
    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(config.NUM_EPOCHS):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, config.WINDOW_SIZE, config.INPUT_SIZE).to(config.DEVICE)
            output = model(seq)
            loss = criterion(output, label.to(config.DEVICE))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, config.NUM_EPOCHS, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    model_file = config.MODEL_DIR + '/' + log + '.pt'
    torch.save(model.state_dict(), model_file)
    writer.close()
    print('Finished Training')
    return model_file
