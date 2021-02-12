import os
import time
import torch
from torch import optim, nn
from data_helper import get_iter
from models import ESIM
from tools import train_for_epoch, evaluate_model, epoch_time


def training(N_EPOCH, model, train_iter, dev_iter, device):
    optimizer = optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    best_epoch = 0
    best_dev_loss = float('inf')

    for epoch in range(N_EPOCH):
        start_time = time.time()
        train_loss, train_acc = train_for_epoch(model, train_iter, optimizer, criterion)
        dev_loss, dev_acc = evaluate_model(model, dev_iter, criterion)
        end_time = time.time()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            if not os.path.exists('results'):
                os.mkdir('results')
            torch.save(model.state_dict(), 'results/ESIM_model.pt')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print("Epoch: %02d | Epoch Time: %02dm %02ds" % (epoch, epoch_mins, epoch_secs))
        print("\tTrain Loss: %6.3f | Train Acc: %6.2f%%" % (train_loss, train_acc*100))
        print("\tDev   Loss: %6.3f | Dev   Acc: %6.2f%%" % (dev_loss, dev_acc * 100))

    print('-' * 60)
    print("Best epoch %02d" % best_epoch)
    print('-' * 60)


def evaluate(model, test_iter, device):
    model.load_state_dict(torch.load('results/ESIM_model.pt'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_iter, criterion)
    print("Test  Loss: %6.3f | Test  Acc: %6.2f%%" % (test_loss, test_acc * 100))


if __name__ == '__main__':
    vector_path = 'F:/DATASET/Glove/glove.6B/glove.6B.300d.txt'
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (text_field, label_field), (train_iter, dev_iter, test_iter) = get_iter(vector_path=vector_path,
                                                                            train_path='./data/snli_1.0_train.csv',
                                                                            dev_path='./data/snli_1.0_dev.csv',
                                                                            test_path='./data/snli_1.0_test.csv',
                                                                            file_format='csv',
                                                                            batch_size=32,
                                                                            torch_device=torch_device)

    model = ESIM(vocab_len=len(text_field.vocab),
                 wordvc_dim=300,
                 hidden_dim=300,
                 output_dim=3,
                 fine_tune=True,
                 weight_matrix=text_field.vocab.vectors,
                 pretrained=True,
                 dropout=0.5)
    model.to(torch_device)

    training(17, model, train_iter, dev_iter, torch_device)
    evaluate(model, test_iter, torch_device)
