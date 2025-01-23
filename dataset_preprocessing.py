from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, DistributedSampler
import os
import pickle
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoConfig, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import subprocess

NUM_SEQ, MIN_SEQ_LENGTH = 4, 512


class PreprocessingDatasetTemplate(Dataset):
    def __init__(self, data_path, tokenizer, shuffle=True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.memmap_path = data_path + '.memmap'
        self.memmap_dtype = [('url_token', np.int32, 512),
                             ('text_token', np.int32, (NUM_SEQ, MIN_SEQ_LENGTH)),
                             ('label', np.int16, 2),  # for multi classification, this number is the categories count
                             ('index', 'S1024')]
        if os.path.exists(self.memmap_path):
            print('Loading memmap from disk')
            self.memmap_file = np.memmap(self.memmap_path, dtype=self.memmap_dtype, mode='r')
            print('Loaded memmap from disk')

        # generate memmap if it doesn't exist
        if not os.path.exists(self.memmap_path):
            self.memmap_file = self._load_memmap(self.data_path, self.memmap_path)
        print(f'memmap file length: {len(self.memmap_file)}')
        # print("Creating samples..")       # if you need do further preprocess for the data
        # self.samples = []
        # for i in range(len(self.memmap_file)):
        #     self.samples.append((True, i))
        # if shuffle:
        #     random.shuffle(self.samples)
        # print("Done shuffling samples")

    def _load_memmap(self, data_path, memmap_path):
        print('Count the data number of memmap')

        def wccount(file_path):
            out = \
            subprocess.Popen(['wc', '-l', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
            return int(out.partition(b' ')[0])

        shape = (wccount(data_path),)

        # Create memmap to cache processed(tokenized) data
        memmap_arr = np.memmap(memmap_path, dtype=self.memmap_dtype, shape=shape, mode='w+')
        for idx, sample in enumerate(open(data_path, 'rb')):
            sample = sample.decode('utf-8').strip().split('\t')
            url, html_text, label, index = sample
            if len(html_text) < 1:
                html_text = '\ufeff'
            raw_label = 1 if float(label) > 0 else 0
            assert url is not None and label is not None and html_text is not None, "Url, Label, HtmlText should not be None!"
            url_token = torch.tensor(
                [self.tokenizer.cls_token_id] + self.tokenizer.encode(url, add_special_tokens=False, max_length=511,
                                                                      padding="max_length", truncation=True))
            if html_text.startswith('\ufeff'):
                full_text_token = torch.tensor(self.tokenizer.encode(html_text, add_special_tokens=True,
                                                                     max_length=2048, padding="max_length",
                                                                     truncation=True))
            else:
                full_text_token = torch.tensor(
                    [self.tokenizer.cls_token_id] + self.tokenizer.encode(html_text, add_special_tokens=True,
                                                                          max_length=2047, padding="max_length",
                                                                          truncation=True))
            label = self._generate_label(raw_label)

            text_token = full_text_token.reshape(NUM_SEQ, MIN_SEQ_LENGTH)
            memmap_arr[idx]['url_token'] = url_token
            memmap_arr[idx]['text_token'] = text_token
            memmap_arr[idx]['label'] = torch.tensor(label, dtype=torch.float16)
            memmap_arr[idx]['index'] = index.encode('utf-8')
            if idx % 10000 == 0:
                print(f'Processed {idx} samples')
        memmap_arr.flush()
        return memmap_arr

    # it's usefull when doing mutiple classification
    def _generate_label(self, raw_label, categories=2):
        res_label = np.zeros(2)
        if raw_label != 0:
            res_label[raw_label] = raw_label
        return res_label

    def __getitem__(self, index):
        url_token, text_token, label, data_index = self.memmap_file[index]
        return url_token, text_token, label, data_index.decode('utf-8')

    def __len__(self):
        return len(self.memmap_file)


"""
****************************************************** Unit tests ******************************************************
"""


def test_preprocessing_dataset_template():
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    dataset = PreprocessingDatasetTemplate('./tmp_file_for_local_test.tsv', tokenizer)
    for index in range(len(dataset)):
        url_token, text_token, label, index = dataset[index]
        print(f'url_token: {url_token}')
        print(f'text_token: {text_token}')
        print(f'label: {label}')
        print(f'index: {index}')


if __name__ == '__main__':
    # Test PreprocessingDatasetTemplate
    test_preprocessing_dataset_template()