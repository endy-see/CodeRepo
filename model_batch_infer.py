from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import pickle
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoConfig, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import datetime
import sys
import time
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support
import subprocess
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from dataset_preprocessing import PreprocessingDatasetTemplate
from model_customize import TeacherModelV2
from config import try_model_names
import argparse


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2,3" if torch.cuda.is_available() else "cpu")  # specify the GPU id's


def load_model(model, resume_from_checkpoint):
    checkpoint = torch.load(resume_from_checkpoint)
    # if keys start from _orig_mod., then remove _orig_mod. from keys
    # Need to debug, if no necessary, then delete it
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[10:]] = checkpoint['model'][key]
            del checkpoint['model'][key]

    # create a copy of checkpoint model state
    new_state_dict = checkpoint['model'].copy()
    model.load_state_dict(new_state_dict, strict=False)
    print(f'--->>> model loaded from checkpoint path: {resume_from_checkpoint}')

    # load prev iter num
    if 'iter_num' in checkpoint:
        prev_iter_num = int(checkpoint['iter_num'])
        print(f'prev_iter_num loaded: {prev_iter_num}')
    # load prev num samplers seen per class
    if 'num_samples_seen_per_category' in checkpoint:
        prev_num_samples_seen_per_category = checkpoint['num_samples_seen_per_category']
        print(f'prev_num_samples_seen_per_category loaded: {prev_num_samples_seen_per_category}')
    return model


def infer_batch(model, tokenizer, infer_file, batch_size, save_file):
    data_set = PreprocessingDatasetTemplate(infer_file, tokenizer)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    data_index_list = []
    score_list = []
    label_list = []
    for batch in enumerate(data_loader):
        url_token, text_token, label, data_index = batch
        url_token = url_token.to(device)
        text_token = text_token.to(device)
        label = label.to(device)

        with torch.no_grad():
            model_score, model_embeddings = model(url_token, text_token)
            outputs = F.sigmoid(model_score)
        score_list.append(outputs)
        label_list.extend(label[:,-1].tolist())
        data_index_list.extend(data_index)
    final_scores = torch.cat(score_list, dim=0)
    pred_scores = [pred_list[-1] for pred_list in final_scores]
    pr, re, thr = precision_recall_curve(label_list, pred_scores)
    pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
    auprc1 = auc(pr_curve[0], pr_curve[1])
    print('auprc1: ', auprc1)
    auprc2 = auc(re, pr)
    print('auprc2: ', auprc2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--infer_file', type=str)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(try_model_names['multilingual_bert'])
    model = TeacherModelV2(try_model_names['multilingual_bert'], url_layer_num=1, text_layer_num=1, combine_layer_num=1)
    model = load_model(model, args.resume_from_checkpoint)
    model = model.to(device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'infer_output.txt')
    infer_batch(model, tokenizer, args.infer_file, args.batch_size, save_path)
