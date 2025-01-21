import os
import shutil
import datetime
import argparse
import numpy as np
from sklearn import metrics
import gzip
import torch
import torch.nn as nn
import random
import pdb

print(
    "################################################ All imports done  ##################################################")

argParse = argparse.ArgumentParser()
argParse.add_argument("--device", type=int, default=0)
argParse.add_argument("--epochs", type=int, default=5)
argParse.add_argument("--batchSize", type=int, default=32)
argParse.add_argument("--buffer_Size", type=int, default=10240)
argParse.add_argument("--logbatchSize", type=int, default=10000)
argParse.add_argument("--seed_val", type=int, default=-1)
argParse.add_argument("--model_path", type=str, default="model.pt")
argParse.add_argument("--train_file", type=str, default="train.tsv")
argParse.add_argument("--test_file", type=str, default="test.tsv")
argParse.add_argument("--exp_name", type=str, required=True)
argParse.add_argument("--file_directory", type=str, required=True)
argParse.add_argument("--lr", type=float, default=0.001)
argParse.add_argument("--drop_prob", type=float, default=0.3)
argParse.add_argument("--drop_url_prob", type=float, default=0)
argParse.add_argument("--threshold", type=float, default=-1)
argParse.add_argument("--features_start_index", type=int, default=0)
argParse.add_argument("--features_end_index", type=int, default=3)
argParse.add_argument("--feature_indices", type=str, default="")
argParse.add_argument("--enable_shuffle", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Shuffle flag, By Default No Shuffle
argParse.add_argument("--enable_finetune", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_instance_weighting", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_cbs_score", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_cbs_emb", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_cbs_v5", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_html", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_trusted_score", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--enable_attention", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--eval_only", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--seif", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--exclude_sen", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--separate_flag", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--small_model", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--as_is", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argParse.add_argument("--random_seeding", default=False, action='store_true',
                      help='Bool type')  ## Enable/disable Finetune flag, By Default No Finetune
argList = argParse.parse_args()

# if argList.eval_only:
#   assert "test" in argList.test_file
# else:
#   assert "test" not in argList.test_file

if argList.enable_attention:
    argList.enable_cbs_score = True
    argList.enable_cbs_emb = True
    argList.enable_cbs_v5 = True
    argList.enable_html = True

if argList.random_seeding:
    # seed_val = torch.initial_seed() % 1000000
    seed_val = torch.random.initial_seed()
    if seed_val < 0:
        seed_val = -seed_val
    if seed_val > 2 ** 32 - 1:
        seed_val = seed_val % (2 ** 32 - 1)
else:
    seed_val = argList.seed_val

# drop_start_index = argList.drop_start_index #inclusive
# drop_end_index = argList.drop_end_index #inclusive
# drop_criterion_index = argList.drop_criterion_index
# print(f'Dropping Embedding flag : {argList.drop_embedding}')
print("+++++++++++++++++++ USING SEED +++++++++++++++++++++++++++++++", seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
# torch.backends.cudnn.deterministic = True


device = torch.device(argList.device if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

start = argList.features_start_index
end = argList.features_end_index

model_path = argList.model_path
directory = argList.file_directory

num_epochs = argList.epochs  ## int(sys.argv[1])     # The number of times entire dataset is trained
batch_size = argList.batchSize  # The size of input data took for one iteration
learning_rate = argList.lr  # The speed of convergence
drop_prob = argList.drop_prob

crit_reduction = 'none' if argList.enable_instance_weighting else 'mean'
filename_prepend = argList.exp_name
'''
if argList.enable_cbs:
    filename_prepend = 'cbs'
elif argList.enable_finetune: 
    filename_prepend = 'finetuned' 
else:
    filename_prepend = 'trained' 
'''
print("Mode : ", filename_prepend)

# The number of nodes at the hidden layer
if argList.small_model:
    hidden_size_1 = 128
    hidden_size_2 = 64
    hidden_size_3 = 32
    hidden_size_4 = 8
else:
    hidden_size_1 = 500
    hidden_size_2 = 500
    hidden_size_3 = 500
    hidden_size_4 = 10

print(
    "############################################## All Arguments parsed  ################################################")
print(f"directory: {directory}")
print(f"Shuffle mode: {argList.enable_shuffle}")


#
# ================================================================
# choose model type

def run_cross_entropy_model(model, input_data, ground_truth, criterion, data_weights=None):
    criterion_pred = model(input_data)[-2]
    y_pred = torch.nn.functional.softmax(criterion_pred, dim=1)
    if argList.enable_instance_weighting:
        loss_batch = criterion(criterion_pred, ground_truth)
        loss = torch.mean(loss_batch * data_weights)
    else:
        loss = criterion(criterion_pred, ground_truth)
    return y_pred[:, 1], loss


def run_cross_entropy_reconstruction_model(model, input_data, ground_truth, criterion, data_weights=None):
    output_all = model(input_data)
    criterion_pred = output_all[-3]
    reconstruction_crit_pred = output_all[-1]
    y_pred = torch.nn.functional.softmax(criterion_pred, dim=1)
    l1loss = nn.L1Loss(reduction=crit_reduction)
    if argList.enable_instance_weighting:
        crit_loss_batch = criterion(criterion_pred, ground_truth)
        crit_loss = torch.mean(crit_loss_batch * data_weights)
        reconstruction_loss_batch = l1loss(reconstruction_crit_pred, input_data)
        reconstruction_loss = torch.mean(torch.transpose(reconstruction_loss_batch, 0, 1) * data_weights)
        loss = (5000.0 * crit_loss) + (1.0 * reconstruction_loss)
    else:
        crit_loss = criterion(criterion_pred, ground_truth)
        reconstruction_loss = l1loss(reconstruction_crit_pred, input_data)
        loss = (5000.0 * crit_loss) + (1.0 * reconstruction_loss)

    return y_pred[:, 1], loss


def run_binary_cross_entropy_model(model, input_data, ground_truth, criterion, data_weights=None):
    criterion_pred = model(input_data)[-1][:, 0]
    formatted_gt = torch.as_tensor(ground_truth, dtype=torch.float32)
    if argList.enable_instance_weighting:
        loss_batch = criterion(criterion_pred, formatted_gt)
        loss = torch.mean(loss_batch * data_weights)
    else:
        loss = criterion(criterion_pred, formatted_gt)
    return criterion_pred, loss


model_type = 'CROSS_ENTOPY'
# model_type = 'CROSS_ENTROPY_RECONSTRUCTION'
# model_type = 'BINARY_CROSS_ENTROPY'

if model_type == 'CROSS_ENTOPY':
    from fc_model import Net
    from model import AttentionNet

    run_model = run_cross_entropy_model
    criterion = nn.CrossEntropyLoss()
    num_classes = 2
    # save_directory = 'experiments/missing_features_fixed/Weighted_Fine_Tune_3_7_noH2_query_urlweighting'
    save_directory = 'experiments/{}'.format(filename_prepend)
elif model_type == 'CROSS_ENTROPY_RECONSTRUCTION':
    from model_reconstruction import Net

    run_model = run_cross_entropy_reconstruction_model
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([.3, .7]).to(device), reduction=crit_reduction)
    num_classes = 2
    save_directory = 'experiments/missing_features_fixed/Weighted_Fine_Tune_3_7_Reconstruction_7_3_noH2_query_urlweighting'
elif model_type == 'BINARY_CROSS_ENTROPY':
    from model_binary import Net

    run_model = run_binary_cross_entropy_model
    criterion = nn.BCELoss(reduction=crit_reduction).to(device)
    num_classes = 1
    save_directory = 'experiments/missing_features/Single_Label_Binary_Fine_Tune_noH2'

if not argList.eval_only:
    shutil.rmtree(save_directory, ignore_errors=True)
    os.makedirs(save_directory)


#
# ================================================================
# data loader

class EmployeeStreamLoader():
    def __init__(self, fn, bat_size, buff_size, shuffle=True, seed=0, apply_instance_weight=False, drop_url=False):
        if buff_size % bat_size != 0:
            raise Exception("buff_size must be evenly div by bat_size")

        self.bat_size = bat_size
        self.buff_size = buff_size
        self.shuffle = shuffle
        self.apply_instance_weight = apply_instance_weight

        self.rnd = np.random.RandomState(seed)

        self.ptr = 0  # points into x_data and y_data
        # self.fin = gzip.open(fn)  # line-based text file
        self.fin = open(fn)

        self.drop_url = drop_url

        self.the_buffer = []  # list of numpy vectors
        self.buffer_query = []
        self.xy_mat = None  # NumPy 2-D version of buffer
        self.x_data = None  # predictors as Tensors
        self.y_data = None  # targets as Tensors
        self.w_data = None  # instance weights as Tensors
        self.reload_buffer()
        print(self.xy_mat.shape)

    def reload_buffer(self):
        self.ptr = 0
        self.the_buffer = []
        self.buffer_query = []
        for ct in range(self.buff_size):
            # line = self.fin.readline().decode('utf-8')
            line = self.fin.readline()
            if line == "":
                self.fin.seek(0)
                return -1  # reached EOF
            else:
                '''
                Url	string
                Class string
                Features	string
                SpamLabel	float
                Query	string
                HostKey	string
                DRScore	double?
                TrustedScore	int?
                Label	string
                Rand	int
                IndexStatus	string
                QuerySampleDate	string
                JudgeSpamRate	double
                '''
                try:
                    split_line = line.split('\t')
                    if split_line[1] != '':
                        # The last dimension of input is label
                        '''
                        if self.apply_instance_weight:
                            trusted_score = int(split_line[6]) if split_line[6] else 0
                            dr_score = float(split_line[5]) if split_line[5] else 0
                            judge_spam_rate = float(split_line[11].strip()) if split_line[11].strip() else 0
                            instance_weight = self.weight_instance(query_sample_date=split_line[10],
                                                                judge_spam_rate=judge_spam_rate,
                                                                index_status=split_line[9],
                                                                trusted_score=trusted_score,
                                                                dr_score=dr_score,
                                                                other_label=split_line[7],
                                                                weight=10.0)
                            features_combined = f'{split_line[1]}|{split_line[2]}|{instance_weight}'
                        else:
                            features_combined = f'{split_line[2]}|{split_line[3]}'
                        '''
                        tokens = []
                        if argList.feature_indices != "":
                            features = split_line[1].replace("!", "|").split("|")
                            for interval in argList.feature_indices.split(","):
                                start, end = interval.split("-")
                                if argList.drop_url_prob > 0 and features[601] == '1':
                                    if np.random.uniform() < argList.drop_url_prob:
                                        features[345:602] = ['0'] * len(features[345:602])
                                tokens.extend(features[int(start) - 1:int(end)])
                        else:
                            if argList.drop_url_prob > 0 and split_line[5] == '1' and self.drop_url:
                                if np.random.uniform() < argList.drop_url_prob:
                                    split_line[4] = "|".join(['0'] * 256)
                                    split_line[5] = "0"
                            if not argList.seif:
                                tokens = split_line[1:6]
                                if argList.enable_cbs_score:
                                    if argList.enable_cbs_emb:
                                        tokens.extend([split_line[x] for x in (8, 9, 10)])
                                    else:
                                        tokens.extend([split_line[x] for x in (8, 10)])
                                elif argList.enable_cbs_emb:
                                    tokens.extend([split_line[x] for x in (9, 10)])

                                if argList.enable_cbs_v5:
                                    tokens.extend([split_line[x] for x in (11, 12, 13)])

                                if argList.enable_html:
                                    tokens.extend([split_line[x] for x in (14, 15)])
                            else:
                                tokens = split_line[1:2]
                                if argList.exclude_sen:
                                    tokens = tokens[0].split('|')
                                    tokens = ['|'.join(tokens[:100] + tokens[484:])]
                                if argList.separate_flag:
                                    if argList.enable_cbs_score:
                                        if argList.enable_cbs_emb:
                                            tokens.extend([split_line[x] for x in (2, 3, 4, 5)])
                                        else:
                                            tokens.extend([split_line[x] for x in (2, 3)])
                                    elif argList.enable_cbs_emb:
                                        tokens.extend([split_line[x] for x in (4, 5)])

                                    if argList.enable_cbs_v5:
                                        tokens.extend([split_line[x] for x in (6, 7, 8)])

                                    if argList.enable_html:
                                        tokens.extend([split_line[x] for x in (9, 10)])

                                    if argList.enable_trusted_score:
                                        tokens.extend([split_line[x] for x in (11, 12)])
                                else:
                                    if argList.enable_cbs_score:
                                        if argList.enable_cbs_emb:
                                            tokens.extend([split_line[x] for x in (2, 3, 4)])
                                        else:
                                            tokens.extend([split_line[x] for x in (2, 4)])
                                    elif argList.enable_cbs_emb:
                                        tokens.extend([split_line[x] for x in (3, 4)])

                                    if argList.enable_cbs_v5:
                                        tokens.extend([split_line[x] for x in (5, 6, 7)])

                        tokens.append(split_line[-1])
                        features_combined = '|'.join(tokens)

                        features = np.array([float(x) for x in features_combined.replace('!', '|').split('|')],
                                            dtype=np.float32)

                        self.buffer_query.append(split_line[0])
                        self.the_buffer.append(features)
                except Exception as e:
                    print("error for", split_line, e)

        if len(self.the_buffer) != self.buff_size and not self.drop_url:
            return -2  # buffer was not fully loaded

        if self.shuffle:
            index_list = list(range(len(self.the_buffer)))
            self.rnd.shuffle(index_list)  # use index list to keep the_buffer and buffer_query aligned
            self.the_buffer = [self.the_buffer[i] for i in index_list]
            self.buffer_query = [self.buffer_query[i] for i in index_list]

        self.xy_mat = np.array(self.the_buffer)  # 2-D array
        if self.apply_instance_weight:
            self.x_data = torch.tensor(self.xy_mat[:, :-2], dtype=torch.float32).to(device)
            self.y_data = torch.tensor(self.xy_mat[:, -2], dtype=torch.int64).to(device)
            self.w_data = torch.tensor(self.xy_mat[:, -1], dtype=torch.float32).to(device)
        else:
            self.x_data = torch.tensor(self.xy_mat[:, :-1], dtype=torch.float32).to(device)
            self.y_data = torch.tensor(self.xy_mat[:, -1], dtype=torch.int64).to(device)
        return 0  # buffer successfully loaded

    def weight_instance(self, query_sample_date, judge_spam_rate, index_status,
                        trusted_score, dr_score, other_label, weight):
        split_date = query_sample_date.split('-')
        formatted_date = datetime.datetime(int(split_date[0]), int(split_date[1]), int(split_date[2]))
        date_w = 1 - min(330, abs((datetime.date.today() - formatted_date.date()).days)) / 365.0
        spam_w = 1.0 if judge_spam_rate >= 0.1 else 0.1
        indexed_w = 1.0 if index_status == 'IN' else 0.1
        importance = 1.0
        if judge_spam_rate < 0.1:  # only overweight important hosts when mistakenly labeling them as spam
            if trusted_score >= 20:
                if trusted_score >= 30:
                    importance += 10
                else:
                    importance += 3
            if dr_score >= 3.0:
                if dr_score >= 3.5:
                    importance += 10
                else:
                    importance += 2
            if other_label == 'Trusted':
                importance += 4
        return importance * date_w * spam_w * indexed_w * weight

    def __iter__(self):
        return self

    def __next__(self):  # next batch as a tuple
        res = 0

        # if self.ptr + self.bat_size > self.buff_size:  # reload
        if self.ptr + self.bat_size > len(self.the_buffer):  # reload
            # print(" ** reloading buffer ** ")
            res = self.reload_buffer()
            # 0 = success, -1 = hit eof, -2 = not fully loaded

        if res == 0:
            start_idx = self.ptr
            end_idx = self.ptr + self.bat_size
            x = self.x_data[start_idx:end_idx]
            y = self.y_data[start_idx:end_idx]
            q = self.buffer_query[start_idx:end_idx]
            self.ptr += self.bat_size
            if self.apply_instance_weight:
                w = self.w_data[start_idx:end_idx]
                return (x, y, q, w)
            else:
                return (x, y, q)

        # reached end-of-epoch (EOF), so signal no more
        self.reload_buffer()  # prepare for next epoch
        raise StopIteration


#
# ================================================================
# test function


def test_loaded_model(type_directory, path, criterion):
    print(f"Test loaded model path: {path}")

    # load model
    model = torch.load(path).to(device)
    model.eval()

    # make directories
    shutil.rmtree(type_directory, ignore_errors=True)
    os.makedirs(type_directory)
    os.makedirs(f'{type_directory}/False_positives')
    os.makedirs(f'{type_directory}/False_negatives')
    os.makedirs(f'{type_directory}/True_positives')
    os.makedirs(f'{type_directory}/True_negatives')

    test_loss = []
    threshold_prs = {}
    write_th = 0.5
    threshold_granularity = 100
    threshold_steps = np.linspace(0, 1, threshold_granularity + 1)

    with open(f'{type_directory}/False_positives/FP{write_th}.tsv', 'w') as fp_file, \
            open(f'{type_directory}/False_negatives/FN{write_th}.tsv', 'w') as fn_file, \
            open(f'{type_directory}/True_positives/TP{write_th}.tsv', 'w') as tp_file, \
            open(f'{type_directory}/True_negatives/TN{write_th}.tsv', 'w') as tn_file, \
            open(f"{type_directory}/predictions.tsv", 'w') as pred_file:

        for (b_idx, batch) in enumerate(test_sdl):
            # load data
            # input_data = batch[0][:,start:end].to(device)
            input_data = batch[0].to(device)
            ground_truth = batch[1].to(device)
            batch_queries = batch[2]
            data_weights = batch[3].to(device) if argList.enable_instance_weighting else None

            # run model
            y_pred, step_loss = run_model(model, input_data, ground_truth, criterion, data_weights)
            test_loss.append(step_loss.item())

            # collect stats
            for i in range(y_pred.shape[0]):
                line = f'{batch_queries[i]}\t{y_pred[i].item()}\t{ground_truth[i].item()}\n'
                label = ground_truth[i].cpu().numpy()
                prediction = y_pred[i].item()

                for th in threshold_steps:
                    threshold_prs.setdefault(th, {'tn': 0, 'fn': 0, 'tp': 0, 'fp': 0})
                    if prediction <= th:
                        if label == 0:
                            threshold_prs[th]['tn'] += 1
                            if write_th == th: tn_file.write(line)
                        else:
                            threshold_prs[th]['fn'] += 1
                            if write_th == th: fn_file.write(line)
                    else:
                        if label == 0:
                            threshold_prs[th]['fp'] += 1
                            if write_th == th: fp_file.write(line)
                        else:
                            threshold_prs[th]['tp'] += 1
                            if write_th == th: tp_file.write(line)

                pred_file.write(line)

    # calculate metrics for thresholds
    threshold_metrics = {}
    trp = []
    for th, vals in threshold_prs.items():
        tp = vals['tp']
        fp = vals['fp']
        tn = vals['tn']
        fn = vals['fn']
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        f_score = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
        tot = tp + fp + tn + fn
        acc = (tp + tn) / tot
        threshold_metrics[th] = {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'accuracy': acc,
            'total': tot,
        }
        trp.append((th, recall, precision))
    trp = sorted(trp, key=lambda x: x[0])
    _, r, p = zip(*trp)
    pr_auc = metrics.auc(r, p)

    # record epoch metrics
    with open(f'{type_directory}/PR.txt', 'w') as sum_file:
        sum_file.write(f'PR Area Under Curve: {pr_auc}\n\n')
        sum_file.write("THRESHOLD\tPRECISION\tRECALL\tF-SCORE\tACCURACY\tTP\tTN\tFP\tFN\tTOTAL\n")
        best_f = -1
        best_th = 0
        for th in threshold_steps:
            th_prs = threshold_prs[th]
            th_m = threshold_metrics[th]
            line = f"{th:.02f}\t{th_m['precision']:.05f}\t{th_m['recall']:.05f}\t{th_m['f_score']:.05f}\t{th_m['accuracy']:.05f}\t{th_prs['tp']}\t{th_prs['tn']}\t{th_prs['fp']}\t{th_prs['fn']}\t{th_m['total']}\n"
            sum_file.write(line)
            if th_m['f_score'] > best_f:
                best_f = th_m['f_score']
                best_th = th

    # print best f score metrics
    print(f"\n++AUPRC: {pr_auc:.05f}++")
    print(f"---- Best Threshold: {best_th:.05f}")
    print(f"F Score: {threshold_metrics[best_th]['f_score']:.05f}")
    print(f"Precision: {threshold_metrics[best_th]['precision']:.05f}")
    print(f"Recall: {threshold_metrics[best_th]['recall']:.05f}")
    print(f"Accuracy: {threshold_metrics[best_th]['accuracy']:.05f}")
    print(f"Total test data size for summary: {threshold_metrics[best_th]['total']}")
    print(
        f"TP: {threshold_prs[best_th]['tp']}, TN: {threshold_prs[best_th]['tn']}, FP: {threshold_prs[best_th]['fp']}, FN: {threshold_prs[best_th]['fn']}")

    print(f"---- At Threshold .5")
    print(f"F Score: {threshold_metrics[.5]['f_score']:.05f}")
    print(f"Precision: {threshold_metrics[.5]['precision']:.05f}")
    print(f"Recall: {threshold_metrics[.5]['recall']:.05f}")
    print(f"Accuracy: {threshold_metrics[.5]['accuracy']:.05f}")
    print(f"Total test data size for summary: {threshold_metrics[.5]['total']}")
    print(
        f"TP: {threshold_prs[.5]['tp']}, TN: {threshold_prs[.5]['tn']}, FP: {threshold_prs[.5]['fp']}, FN: {threshold_prs[.5]['fn']}")
    print(f"-----\n")

    return threshold_metrics[.5]['precision'], threshold_metrics[.5]['recall'], np.mean(test_loss), pr_auc


#
# ================================================================
# main

test_sdl = EmployeeStreamLoader(argList.test_file,
                                batch_size,
                                argList.buffer_Size,
                                shuffle=False,
                                apply_instance_weight=argList.enable_instance_weighting)
if not argList.eval_only:
    emp_ldr = EmployeeStreamLoader(argList.train_file,
                                   batch_size,
                                   argList.buffer_Size,
                                   shuffle=argList.enable_shuffle,
                                   drop_url=True,
                                   apply_instance_weight=argList.enable_instance_weighting)
print(
    "########################################## All Dataloaders Initialized  #############################################")

input_size = test_sdl.x_data.size(1)  # The image size = 28 x 28 = 784
print(f"input size: {input_size}")
# load model

if argList.enable_attention:
    net = AttentionNet(128, 128)
else:
    net = Net(input_size,
              num_classes,
              hidden_size_1,
              hidden_size_2,
              hidden_size_3,
              hidden_size_4,
              drop_prob)
if argList.enable_finetune:
    print("FINETUNING: loading pre-trained model")
    pretrained_model = torch.load(model_path)
    net.load_model_weights(pretrained_model)
    net = torch.load(model_path).to(device)
net.to(device)
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

'''
print("Normal initialization")
for param in net.parameters():
    n = param.size(-1)
    #torch.nn.init.uniform_(param, -np.sqrt(1/n), np.sqrt(1/n))
    torch.nn.init.normal_(param, 0, np.sqrt(2/n))
'''

# set up optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if argList.eval_only:
    model_path = f"experiments/{argList.exp_name}/{argList.exp_name}_best_model.pt"
    p, r, l, pr_auc = test_loaded_model(f"{save_directory}/eval",
                                        model_path,
                                        criterion)
else:
    # test initial model
    torch.save(net, f"{save_directory}/initial_random.pt")
    p, r, l, pr_auc = test_loaded_model(f"{save_directory}/initial_random",
                                        f"{save_directory}/initial_random.pt",
                                        criterion)
    net.train()

    # epoch loop
    np.random.seed(1)
    epoch_losses = []
    test_losses = []
    test_auc = []
    PRS = []
    for epoch in range(num_epochs):
        model_name = f"model_epoch_{epoch}.pt"
        epoch_loss = []
        print(f"\n == Epoch: {epoch} ==")

        # train loop
        for (b_idx, batch) in enumerate(emp_ldr):
            # zero grad
            optimizer.zero_grad()

            # format data
            # input_data = batch[0][:,start:end].to(device)
            input_data = batch[0].to(device)
            labels = batch[1].to(device)
            data_weights = batch[3].to(device) if argList.enable_instance_weighting else None

            # run model
            y_pred, loss = run_model(net, input_data, labels, criterion, data_weights)
            epoch_loss.append(loss.item())
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes

            # log step
            if (b_idx + 1) % argList.logbatchSize == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{b_idx + 1}], Loss: {loss.data.item():.04f}')

        # end of epoch results
        torch.save(net, f"{save_directory}/{filename_prepend}_{model_name}")
        p, r, l, pr_auc = test_loaded_model(f"{save_directory}/{filename_prepend}_epoch_{epoch}",
                                            f"{save_directory}/{filename_prepend}_{model_name}",
                                            criterion)
        test_losses.append(l)
        test_auc.append(pr_auc)
        PRS.append((p, r, pr_auc))
        mean_epoch_loss = np.mean(epoch_loss)
        print(f"SAVED pytorch trained model")
        print(f"Epoch loss: {mean_epoch_loss}")
        epoch_losses.append(mean_epoch_loss)

    # copy over best model
    best_index = np.argmax(test_auc)
    shutil.copyfile(f"{save_directory}/{filename_prepend}_model_epoch_{best_index}.pt",
                    f"{save_directory}/{filename_prepend}_best_model.pt")
    p, r, l, pr_auc = test_loaded_model(f"{save_directory}/output_{filename_prepend}_model",
                                        f"{save_directory}/{filename_prepend}_best_model.pt",
                                        criterion)

    # final prints
    print(f"EPOCH LOSSES: {epoch_losses}")
    print(f"TEST LOSSES: {test_losses}")
    print(f"P/R/AUC's: {PRS}")
    print(f'BEST MODEL EPOCH {best_index} P/R/AUC: {p:.04f} / {r:.04f} / {pr_auc:.04f}')




