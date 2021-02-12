import pandas as pd


def jsonl_to_csv(json_file: str = None, csv_file: str = None, is_test=False, drop_others=True):
    raw_data = pd.read_json(json_file, lines=True)

    df_data = pd.DataFrame()
    df_data['sentence1'] = raw_data['sentence1']
    df_data['sentence2'] = raw_data['sentence2']

    if not is_test:
        label_dic = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 3}
        gold_label = []
        for item in raw_data['gold_label']:
            gold_label.append(label_dic[item])
        df_data['gold_label'] = gold_label
        if drop_others:
            df_data.drop(df_data[df_data['gold_label'] == 3].index, inplace=True)

    df_data.to_csv(csv_file, index=False)


def convert_all():
    jsonl_to_csv(json_file='./data/snli_1.0_train.jsonl', csv_file='./data/snli_1.0_train.csv')
    print('train data converted')

    jsonl_to_csv(json_file='./data/snli_1.0_dev.jsonl', csv_file='./data/snli_1.0_dev.csv')
    print('dev data converted')

    jsonl_to_csv(json_file='./data/snli_1.0_test.jsonl', csv_file='./data/snli_1.0_test.csv')
    print('test data converted')


if __name__ == '__main__':
    convert_all()

    # train_data = pd.read_csv('./data/snli_1.0_train.csv')
    # print(train_data)
    # dev_data = pd.read_csv('./data/snli_1.0_dev.csv')
    # print(dev_data)