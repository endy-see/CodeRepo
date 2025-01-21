# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import BertTokenizer
from dataset_preprocessing import PreprocessingDatasetTemplate
import pdb

try_model_names = {
    'tiny_bert': 'prajjwal1/bert-tiny',
    'multilingual_bert': 'bert-base-multilingual-cased'
}

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def data_preprocessing(data_path, tokenizer):
    # pdb.set_trace()
    dataset = PreprocessingDatasetTemplate(data_path, tokenizer)
    for index in range(len(dataset)):
        item = dataset[index]
        print(item)
    print('666')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    data_path = 'tmp_file_for_local_test.tsv'
    tokenizer = BertTokenizer.from_pretrained(try_model_names['tiny_bert'])
    data_preprocessing(data_path, tokenizer)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
