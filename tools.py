import torch
from torch import nn
from tqdm import tqdm


def accuracy(prediction, label):
    """
    Returns accuracy per batch
    """
    prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
    acc = torch.sum(prediction == label).float() / len(prediction == label)
    return acc


def train_for_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    with tqdm(total=len(iterator), desc="Training Processing", leave=False) as pbar:
        for batch in iterator:
            optimizer.zero_grad()

            input_a = batch.sentence1
            input_b = batch.sentence2
            gold_label = batch.gold_label

            prediction = model(input_a, input_b)
            loss = criterion(prediction, gold_label)
            acc = accuracy(prediction, gold_label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            pbar.update(1)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_model(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            input_a = batch.sentence1
            input_b = batch.sentence2
            gold_label = batch.gold_label

            prediction = model(input_a, input_b)

            loss = criterion(prediction, gold_label)
            acc = accuracy(prediction, gold_label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def predict_labels(model, iterator):
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in iterator:
            input_a = batch.sentence1
            input_b = batch.sentence2

            prediction = model(input_a, input_b)
            prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
            predictions.extend(prediction.tolist())

    return predictions

