import torch
from torch import nn


class ESIM(nn.Module):
    def __init__(self,
                 vocab_len,
                 wordvc_dim,
                 hidden_dim = 300,
                 output_dim = 3,
                 weight_matrix=None,
                 pretrained=False,
                 fine_tune=False,
                 dropout=0.5):
        super(ESIM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_len, wordvc_dim)
        self.pretrained = pretrained
        self.weight_matrix = weight_matrix

        self.fine_tune = fine_tune

        self.encoder = nn.LSTM(input_size=wordvc_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.softmax_a = nn.Softmax(dim=-1)
        self.softmax_b = nn.Softmax(dim=-2)

        self.inference = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.mlp = nn.Linear(8*hidden_dim, output_dim)
        self.act = nn.Tanh()

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        if self.pretrained:
            self.word_embeddings.weight.data.copy_(self.weight_matrix)
            self.word_embeddings.requires_grad_(self.fine_tune)

    def forward(self, input_a, input_b):
        # batch first
        input_a = input_a.transpose(0, 1)
        input_b = input_b.transpose(0, 1)
        # Word embedding
        embeddings_a = self.word_embeddings(input_a)    # [batch_size, seq_len_a, emb_size]
        embeddings_a = self.dropout(embeddings_a)
        embeddings_b = self.word_embeddings(input_b)    # [batch_size, seq_len_b, emb_size]
        embeddings_b = self.dropout(embeddings_b)

        # Input encoding
        encoded_a, _ = self.encoder(embeddings_a)   # [batch_size, seq_len_a, 2 * hidden_size]
        encoded_a = self.dropout(encoded_a)
        encoded_b, _ = self.encoder(embeddings_b)   # [batch_size, seq_len_b, 2 * hidden_size]
        encoded_b = self.dropout(encoded_b)

        # Local inference modeling
        attentions = torch.matmul(encoded_a, encoded_b.transpose(-2, -1))     # [batch_size, seq_len_a, seq_len_b]
        # Local inference collected over sequences
        summation_a = self.softmax_a(attentions)
        summation_a = torch.matmul(summation_a, encoded_b)          # [batch_size, seq_len_a, 2 * hidden_size]

        summation_b = self.softmax_b(attentions).transpose(-2, -1)
        summation_b = torch.matmul(summation_b, encoded_a)          # [batch_size, seq_len_b, 2 * hidden_size]

        # Enhancement of local inference information
        # difference
        diff_a = torch.sub(encoded_a, summation_a)      # [batch_size, seq_len_a, 2 * hidden_size]
        diff_b = torch.sub(encoded_b, summation_b)      # [batch_size, seq_len_b, 2 * hidden_size]
        # element-wise produce
        ewp_a = torch.mul(encoded_a, summation_a)       # [batch_size, seq_len_a, 2 * hidden_size]
        ewp_b = torch.mul(encoded_b, summation_b)       # [batch_size, seq_len_b, 2 * hidden_size]
        # concatenate
        # [batch_size, 4 * seq_len_a, 2 * hidden_size]
        enhancement_a = torch.cat((encoded_a, summation_a, diff_a, ewp_a), dim=-2)
        # [batch_size, 4 * seq_len_b, 2 * hidden_size]
        enhancement_b = torch.cat((encoded_b, summation_b, diff_b, ewp_b), dim=-2)

        #  Inference Composition
        val_a, _ = self.inference(enhancement_a)    # [batch_size, 4 * seq_len_a, 2 * hidden_size]
        val_a = self.dropout(val_a)
        val_b, _ = self.inference(enhancement_b)    # [batch_size, 4 * seq_len_b, 2 * hidden_size]
        val_b = self.dropout(val_b)

        mean_a = torch.mean(val_a, dim=-2)
        max_a, _ = torch.max(val_a, dim=-2)
        mean_b = torch.mean(val_b, dim=-2)
        max_b, _ = torch.max(val_b, dim=-2)

        val = torch.cat((mean_a, max_a, mean_b, max_b), dim=-1)     # [batch_size, 4 * 2 * hidden_size]

        # predict
        output = self.mlp(val)
        output = self.dropout(output)
        output = self.act(output)

        return output


if __name__ == '__main__':
    from data_helper import get_iter

    vector_path = 'F:/DATASET/Glove/glove.6B/glove.6B.300d.txt'
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (text_field, label_field), (train_iter, dev_iter, test_iter) = get_iter(vector_path=vector_path,
                                                                            train_path='./data/snli_1.0_train.csv',
                                                                            dev_path='./data/snli_1.0_dev.csv',
                                                                            test_path='./data/snli_1.0_test.csv',
                                                                            file_format='csv',
                                                                            batch_size=32,
                                                                            torch_device=torch_device)
    model = ESIM(vocab_len=len(text_field.vocab), wordvc_dim=300, hidden_dim=64, output_dim=3, fine_tune=True,
                 weight_matrix=text_field.vocab.vectors, pretrained=True, dropout=0.5)
    model.to(torch_device)
    model.eval()

    for iter in dev_iter:
        # print(iter.sentence1.size())
        print(model(iter.sentence1, iter.sentence2))
        break
