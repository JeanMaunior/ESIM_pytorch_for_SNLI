import torch
from torchtext import data
from torchtext.vocab import Vectors


def get_iter(vector_path: str = None,
             train_path: str = None,
             dev_path: str = None,
             test_path: str = None,
             file_format: str = 'csv',
             batch_size: int = 32,
             torch_device: torch._C = torch.device('cpu')):
    text_field = data.Field(sequential=True)
    label_field = data.Field(sequential=False, use_vocab=False) # labels have been converted to num

    train_dataset = data.TabularDataset(path=train_path, format=file_format, skip_header=True,
                                       fields=[('sentence1', text_field), ('sentence2', text_field),
                                               ('gold_label', label_field)])
    dev_dataset = data.TabularDataset(path=dev_path, format=file_format, skip_header=True,
                                       fields=[('sentence1', text_field), ('sentence2', text_field),
                                               ('gold_label', label_field)])
    test_dataset = data.TabularDataset(path=test_path, format=file_format, skip_header=True,
                                       fields=[('sentence1', text_field), ('sentence2', text_field),
                                               ('gold_label', label_field)])

    # load pretrained GloVe to build vocab
    vectors = Vectors(name=vector_path)
    text_field.build_vocab(train_dataset, vectors=vectors)

    train_iter = data.BucketIterator(train_dataset, batch_size=batch_size, shuffle=True, device=torch_device)
    dev_iter = data.BucketIterator(dev_dataset, batch_size=batch_size, shuffle=True, device=torch_device)
    test_iter = data.Iterator(test_dataset, batch_size=batch_size, train=False, sort=False, device=torch_device)

    return (text_field, label_field), (train_iter, dev_iter, test_iter)


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
    # for iter in dev_iter:
    #     print(iter)
    #     print(iter.sentence1)

    # check the word vector with the downloaded GloVe file
    print(text_field.vocab.vectors[text_field.vocab.stoi['and']])
    print(text_field.vocab.vectors[text_field.vocab.stoi['<unk>']])

