import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, DistributedSampler
import os
import pickle
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import default_data_collator, BertModel, BertTokenizer, AutoModel, AutoConfig, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import datetime
import sys
import time
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support
import subprocess
import torch.nn.init as init
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from v3_dataset import SpamDataset, SpamHtmlDataset, SpamMonoDataset, SpamMonoHtmlDataset, SpamMonoInferenceDataset
from v3_network import SpamStudentModelV2, SpamStudentModelV2WithHtmlFeature, SpamModelV2, SpamStudentModelV2_MBert
from dataset_preprocessing import PreprocessingDatasetTemplate
from tqdm import tqdm
from config import try_model_names
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
# parser.add_argument('--model_id', type=int)
# parser.add_argument('--input_data', type=str)
# parser.add_argument('--max_seq_length', type=int, default=1024)
parser.add_argument('--output_base_dir', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--is_finetune', type=bool)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--train_pos_name', type=str)
parser.add_argument('--train_neg_name', type=str)
parser.add_argument('--valid_pos_name', type=str)
parser.add_argument('--valid_neg_name', type=str)
parser.add_argument('--enc_num_layers', type=int, default=2)
parser.add_argument('--resume_from_checkpoint', type=str, default='./checkpoint_iter_135000')
args = parser.parse_args()

train_sampler = None
output_base_dir = args.output_base_dir
model_save_dir = os.path.join(output_base_dir, 'saved_model_dir')
log_out_dir = os.path.join(output_base_dir, 'log_dir')
writer = SummaryWriter(log_out_dir)

# DDP settings
# use torch.distributed to speed up training
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
# rank = int(os.environ["RANK"])
timeout = timedelta(hours=5)
local_rank = dist.get_rank()
print('local rank:', local_rank, torch.distributed.is_initialized(), world_size)
assert torch.distributed.is_initialized()

gpu_id = int(os.environ["LOCAL_RANK"])  # rank
# gpu_id = dist.get_rank()
print('--->>> gpu_id: ', gpu_id)
# device = 'cuda:3'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = f"cuda:{gpu_id}"
if gpu_id == 0:
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_out_dir, exist_ok=True)

    log_file = open(os.path.join(log_out_dir, 'log.txt'), 'w')

MANUAL_SEED = 42  # meaning of life, the universe, and everything.
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed_all(MANUAL_SEED)  # if you are using multi-GPU.
np.random.seed(MANUAL_SEED)  # Numpy module.
random.seed(MANUAL_SEED)  # Python random module.

# avoid non deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = args.batch_size
NUM_SEQ, MIN_SEQ_LENGTH = 4, 512
enc_num_layers = args.enc_num_layers

max_iters = 80000
grad_clip = 1.0
decay_lr = False
warmup_iters = 4000
lr_decay_iters = 80000
weight_decay = 1e-1
betas = (0.9, 0.95)
learning_rate = 2e-5
min_lr = 1e-5
# gradient_accumulation_steps = 4 if not is_bs_finetune else 1
gradient_accumulation_steps = 1
eval_interval = 1000  # 2000
log_interval = 1000


spam_subtype_mapping = {
    0: "BrandStuffing",
    1: "KeywordStuffing",
    2: "Media",
    3: "MGC",
    4: "IsSpam",
    5: "Empty",
    6: "Other",
    7: "Negative"
}
# create reverse mapping
# spam_subtype_reverse_mapping = {v: k for k, v in spam_subtype_mapping.items()}


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:  # warmup_iters: 4000
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
        # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def load_pretrain_model(model, optimizer):
    checkpoint_path = './checkpoint_iter_135000'
    checkpoint = torch.load(checkpoint_path)
    # if keys start from _orig_mod., then remove _orig_mod. from keys
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[10:]] = checkpoint['model'][key]
            del checkpoint['model'][key]

    # Create a copy of checkpoint model state
    new_state_dict = checkpoint['model'].copy()

    model.load_state_dict(new_state_dict, strict=False)
    print("model loaded from checkpoint path: ", checkpoint_path)

    def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    optimizer_to(optimizer, device)
    print("optimizer loaded from checkpoint path: ", checkpoint_path)

    prev_iter_num = int(checkpoint['iter_num'])  # 之前记录在checkpoint中的信息，如果是要接着finetune时，会有用
    print("prev_iter_num loaded: ", prev_iter_num)

    # load prev_num_samples_seen_per_pipeline
    # if 'num_samples_seen_per_pipeline' in checkpoint:
    #     prev_num_samples_seen_per_pipeline = checkpoint['num_samples_seen_per_pipeline']
    #     print("prev_num_samples_seen_per_pipeline loaded: ", prev_num_samples_seen_per_pipeline)


def eval_binary_classifier_subtypes(model, dataloader, iter_num):
    print("evaluating...")
    model.eval()

    label_list = []
    pred_scores = []

    # num_positives, num_negatives = 0, 0

    for idx, (url_tokens, inputs, labels, pipelines) in enumerate(dataloader):
        # move data to the correct device
        #             print(idx)
        inputs = inputs.to(device)
        labels = labels.to(device)
        url_tokens = url_tokens.to(device)

        # get model outputs
        with torch.no_grad():
            model_score, _ = model(url_tokens, inputs)
            outputs = F.sigmoid(model_score)  # outputs: torch.Size([128, 5])
        pred_scores.extend(outputs.view(-1).tolist())  # len(x): 640
        label_list.extend(labels.view(-1).tolist())  # len(x): 640

        # update num_positives and num_negatives from labels
        # num_positives += int(sum(labels.view(-1).tolist()))  # num_positives: 0
        # num_negatives += int(len(labels.view(-1).tolist()) - sum(labels.view(-1).tolist()))

    # print('calculate pr')
    # for idx in range(5):
    #     indices = []        # 所有正样本所在的索引
    #     neg_indices = []    # 所有负样本所在的索引
    #     # get the indices of the current label from label_list. Every 5th element is the current label.
    #     for i in range(idx, len(label_list), 5):  # 4275, 得到某一类别在label_list中的index
    #         if label_list[i] == 1:
    #             indices.append(i)
    #         elif label_list[i] == 0:
    #             neg_indices.append(i)

    #     # get the corresponding outputs and labels and url
    #     output_list_sub = [output_list[i] for i in indices]     # 把正样本的预测结果都填到output_list_sub的前面
    #     label_list_sub = [label_list[i] for i in indices]       # 把正样本的label都在全部放在label_list_sub的前面

    #     if len(output_list_sub) == 0: # 没有这个类别
    #         print("output_list_sub is empty for label: ", spam_subtype_mapping[idx])
    #         continue

    #     # add an equal number of negatives to the current label. Sample randomly from the negatives.
    #     if neg_indices:
    #         neg_indices_sub = neg_indices
    #         # if eval_mikhail_set or is_bs_finetune:
    #         #     # randomly sample from neg_indices
    #         #     neg_indices_sub = random.sample(neg_indices, min(500 * num_positives, len(neg_indices)))
    #         # elif eval_oct_leakage or eval_april_leakage or eval_august_leakage or eval_october_leakage:
    #         #     neg_indices_sub = neg_indices
    #         # else:
    #         #     neg_indices_sub = random.sample(neg_indices, min(len(output_list_sub) * 4, len(neg_indices))) if not (eval_oct_leakage or eval_mikhail_set or is_bs_finetune) else neg_indices

    #         # output_list_sub.extend([output_list[i] if not is_bs_finetune else output_list[i//5] for i in neg_indices_sub])
    #         # label_list_sub.extend([label_list[i] for i in neg_indices_sub])
    #         output_list_sub.extend([output_list[i] for i in neg_indices_sub])  # 再把所有负样本的预测结果紧贴着output_list_sub中正样本的后面放
    #         label_list_sub.extend([label_list[i] for i in neg_indices_sub])    # 再把所有负样本的真值结果贴着label_list_sub中正样本的后面放（老实说，你这么排顺序没啥用啊，可能对有subtype的时候有用吧。。）

    #     print("num of 1, num of 0 in label_list for label {}: {}, {}".format(spam_subtype_mapping[idx], sum(label_list_sub), len(label_list_sub) - sum(label_list_sub)))

    # Temporary comments
    from sklearn.metrics import average_precision_score
    pr, re, thr = precision_recall_curve(label_list, pred_scores)
    beta = 0.5
    fbeta = (1 + beta ** 2) * pr * re / ((beta ** 2) * pr + re + 0.00000000001)
    index = np.nanargmax(fbeta)
    print("Threshold: ", thr[index])
    print("Fbeta: ", fbeta[index])
    print('auprc: ', average_precision_score(label_list, pred_scores))
    print("Precision: ", pr[index])
    print("Recall: ", re[index])

    pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
    auprc1 = auc(pr_curve[0], pr_curve[1])
    print('auprc1: ', auprc1)
    auprc2 = auc(re, pr)
    print('auprc2: ', auprc2)

    # pr, re, thr = precision_recall_curve(label_list, output_list)

    # pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
    # auprc = auc(pr_curve[0], pr_curve[1])
    # print("auprc for label {}: {}".format(spam_subtype_mapping[idx], auprc))
    # log_file.write(f'auprc for label {spam_subtype_mapping[idx]}: {auprc}\n')
    # print(f"auprc for spam: {auprc}")

    model.train()


class CustomMaskedLoss(nn.Module):
    def __init__(self):
        super(CustomMaskedLoss, self).__init__()
        # weight = [4.0, 4.0, 4.0, 4.0, 7.3]
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight), reduction='none') if is_bs_finetune else nn.BCEWithLogitsLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, label, pipelines):  # student_scores, labels, pipelines
        # create mask for relevant labels
        mask = torch.ones_like(label)

        # -> Exp1 (with subtypes) -> Exp3 (with subtypes & from scratch)
        # Mask subtypes: if pipeline is empty, then mask out all labels except for "IsSpam"
        # for idx, pipeline in enumerate(pipelines):
        #     # if not is_bs_finetune:
        #     # get pipeline type. Position of "empty", rest will be 0.
        #     if sum(pipeline) == 1 and pipeline[spam_subtype_reverse_mapping["Empty"]] == 1:  # spam_subtype_reverse_mapping["Empty"]: 5
        #         assert label[idx, spam_subtype_reverse_mapping["IsSpam"]] == 1
        #         mask[idx, :spam_subtype_reverse_mapping["IsSpam"]] = 0
        #         mask[idx, spam_subtype_reverse_mapping["IsSpam"]] = 1

        # calculate loss and apply mask. output is float, label is int.
        loss = self.loss(output, label)
        loss = loss * mask  # 5 dim

        nonzero_loss = loss[loss != 0]
        return nonzero_loss.mean() if nonzero_loss.nelement() != 0 else torch.tensor(0.0, device=device,
                                                                                     requires_grad=True)


def load_data(data_dir, train_pos_name, train_neg_name, valid_pos_name, valid_neg_name):
    # train_pos_file = '/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/FBVModel/0114TokensForTrainingCbspamPos.tsv'
    # train_neg_file = '/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/FBVModel/0114TokensForTrainingCbspamNeg.tsv'
    # valid_pos_file = '/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/FBVModel/0114TokensForEvalCbspamPos.tsv'
    # valid_neg_file = '/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/FBVModel/0114TokensForEvalCbspamNeg.tsv'
    train_pos_file = os.path.join(data_dir, train_pos_name)
    train_neg_file = os.path.join(data_dir, train_neg_name)
    valid_pos_file = os.path.join(data_dir, valid_pos_name)
    valid_neg_file = os.path.join(data_dir, valid_neg_name)

    # load the data
    train_dataset = SpamDataset(train_pos_file, train_neg_file, shuffle=True)
    valid_dataset = SpamDataset(valid_pos_file, valid_neg_file, shuffle=False)
    print("loaded datasets for training and validation")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_sampler.set_epoch(0)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size // world_size, sampler=train_sampler, pin_memory=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True,
                              num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    print("Loaded {} training samples".format(len(train_dataset)))
    print("Loaded {} validation samples".format(len(valid_dataset)))
    return train_loader, valid_loader, train_sampler


def train(model, optimizer):
    loss_function = CustomMaskedLoss()
    iter_num = 0
    epoch = 0
    # if prev_iter_num and not is_finetune:
    #     iter_num = prev_iter_num

    t0 = time.time()
    train_loader, valid_loader, train_sampler = load_data(args.data_dir, args.train_pos_name, args.train_neg_name,
                                                          args.valid_pos_name, args.valid_neg_name)
    # num_samples_seen_per_pipeline = defaultdict(int)
    # if prev_num_samples_seen_per_pipeline and not is_finetune:
    #     num_samples_seen_per_pipeline = prev_num_samples_seen_per_pipeline
    print('--------->>>>>> model save path: ', model_save_dir)
    while True:  # 啥时候停，全看训练过程中eval的结果，有效果更好的就手动停止
        # if epoch > 50:
        #     exit(0)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        print(f'total steps: {len(train_loader)}')
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}'):
            # determine the learning rate for this iteration
            if decay_lr:
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = learning_rate

            if gpu_id == 0 and iter_num % eval_interval == 0 and iter_num != 0:
                print('model save path: ', model_save_dir)
                log_file.write(f'model save path: {model_save_dir}\n')
                eval_binary_classifier_subtypes(model, valid_loader, iter_num)

                raw_model = model.module if hasattr(model, "module") else model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'lr': 'lr',
                    # 'num_samples_seen_per_pipeline': num_samples_seen_per_pipeline,
                }

                # checkpoint_name = f"checkpoint_iter_{iter_num}_step{step}_epoch{epoch}_zhym"
                # print("saving checkpoint to path: ", os.path.join(model_save_dir, checkpoint_name))
                # torch.save(checkpoint, os.path.join(model_save_dir, checkpoint_name))
                # reset to train mode
                model.train()
                # train
            url_tokens, data, labels, pipelines = batch
            if True:
                # if ddp:
                model.require_backward_grad_sync = (iter_num % gradient_accumulation_steps == 0)
            # with ctx:
            scores, _ = model(url_tokens.to(device), data.to(device))
            labels = labels.to(device)
            loss = loss_function(scores, labels.float(), pipelines)
            # if scaler is not None:
            #     # backward pass, with gradient scaling if training in fp16
            #     scaler.scale(loss).backward()

            #     if iter_num % gradient_accumulation_steps == 0: # how many batches to accumulate gradients for. gradient_accumulation_steps: 4
            #         # clip gradients
            #         if grad_clip != 0:  # grad_clip: 1
            #             scaler.unscale_(optimizer)
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            #         # step the optimizer
            #         scaler.step(optimizer)
            #         scaler.update()

            #         # flush the gradients
            #         optimizer.zero_grad(set_to_none=False)
            # else:
            loss.backward()
            if iter_num % gradient_accumulation_steps == 0:
                if grad_clip != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)
                # log training loss
            if gpu_id == 0 and iter_num % log_interval == 0:
                lossf = loss.item()  # loss: tensor(0.0402, device='cuda:0', grad_fn=<AddBackward0>)
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

                # log to tensorboard
                writer.add_scalar('train/loss', loss.item(), iter_num)
                writer.add_scalar('train/lr', lr, iter_num)

                # try:
                #     num_examples_seen = iter_num * batch_size * world_size   # world_size: 1
                #     writer.add_scalar('train/num_examples_seen', num_examples_seen, iter_num)
                # except:
                #     print("Error logging weights to tensorboard for iter_num: ", iter_num)

                # # log number of samples seen per pipeline
                # for idx, num_samples in num_samples_seen_per_pipeline.items():
                #     writer.add_scalar(f'train/num_samples_seen_per_pipeline/{spam_subtype_mapping[idx]}', num_samples, iter_num)
                #     print(f"iter {iter_num}: num_samples_seen on 1 gpu for {spam_subtype_mapping[idx]}: {num_samples}")
            iter_num += 1
            # if iter_num >= max_iters:
            #     return
        epoch += 1
    log_file.close()


def main():
    # set tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # MODEL_NAME = 'microsoft/deberta-v3-xsmall'
    # average_embeddings = True

    # DDP settings
    # backend = 'nccl' # 'nccl', 'gloo', etc.
    # ddp = True
    # print(f"ddp: {ddp}")
    # world_size = torch.cuda.device_count() if ddp else 1 # how many processes are there in total
    # if ddp:
    #     from datetime import timedelta
    #     timeout = timedelta(hours=5)
    #     init_process_group(backend=backend, timeout=timeout)
    #     gpu_id = int(os.environ["LOCAL_RANK"])
    #     device = f"cuda:{gpu_id}"
    # else:
    #     gpu_id = 3 # gpu_id 0 means this is the (single) master process, basically

    # if gpu_id == 0:
    #     os.makedirs(out_dir, exist_ok=True)

    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # model= nn.DataParallel(model,device_ids = [0,1,2,3])       需要验证
    # teacher_model = nn.DataParallel(teacher_model,device_ids = [0,1,2,3])

    # data related
    # dtype = 'float32'
    # initialize a GradScaler if data type is float16 else no-op.
    # scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16')) if dtype == 'float16' else None
    scaler = None
    # note: float16 data type will automatically use a GradScaler
    # device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # from contextlib import nullcontext
    # ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

    # if resume training, load provided checkpoint.
    # prev_iter_num = 0
    # prev_num_samples_seen_per_pipeline = None
    # resume_from_checkpoint=False

    # main: load model
    model = SpamStudentModelV2(enc_num_layers=int(enc_num_layers), combine_num_layers=1)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas)

    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    model.cuda()

    is_finetune = args.is_finetune
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        load_pretrain_model(model, optimizer)

    train(model, optimizer)
    exit(0)


if __name__ == '__main__':
    main()

# torch --standalone --nproc_per_node=4 train_v3_with_subtype.py --tensorboard_dir tensorboard_dir --freeze_encoder $freeze_encoder --enc_num_layers $enc_num_layers --average_embeddings $average_embeddings --temperature $temperature --alpha $alpha
# --standalone: 以独立模式运行，不依赖于分布式训练集群或多节点设置，一般适用于在单个节点上进行训练
# --nproc_per_node=4: 指定在每个节点上使用的进程数为4，这进一步强调了是在单进程条件下进行训练
# 此命令使用 torch 以 --standalone 模式运行 train_v3_with_subtype.py 脚本，这通常意味着它将在单个节点上运行，不涉及分布式计算集群

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 60001 --nproc_per_node=4 train_v3_optimized.py

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 torchrun --nnodes 1 --nproc_per_node 16 train_script_optimized_e5_8.py --load_from
# 这里使用 torch run 命令，这是 PyTorch 中用于分布式训练的命令
# --nnodes 1 表示使用的节点数量为 1，即仍然是在单个节点上进行训练，类似于命令 1 中的单机环境
# --nproc_per_node 16 表示在这个单个节点上会启动 16 个进程。与命令 1 不同的是，使用 torch run 可能意味着使用更高级别的分布式训练 API 或机制，尽管在节点数量上都是 1，但它可能利用更现代或更方便的分布式训练框架，提供了更灵活的进程管理和通信机制。
