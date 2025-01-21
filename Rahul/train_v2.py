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
from transformers import BertModel, BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import datetime
import sys
import time
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

# data directory
data_dir = '/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/'

# set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

NUM_SEQ, MIN_SEQ_LENGTH = 4, 512
# create enum of mapping
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
spam_subtype_reverse_mapping = {v: k for k, v in spam_subtype_mapping.items()}


class SpamDataset(Dataset):
    def __init__(self, positive_file, negative_file, shuffle=True):
        # Store the paths to the positive and negative files and compute offsets
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        self.pipeline_counts = defaultdict(int)
        # if pipeline_counts exists, load it from disk.
        if os.path.exists(data_dir + f"{self.positive_file}_{self.negative_file}_pipeline_counts.pkl"):
            with open(
                    data_dir + f"{self.positive_file.split('/')[-1]}_{self.negative_file.split('/')[-1]}_pipeline_counts.pkl",
                    'rb') as f:
                self.pipeline_counts = pickle.load(f)

        self.memmap_positive_path = self.positive_file + '.memmap'
        self.memmap_negative_path = self.negative_file + '.memmap'
        # tokens is of shape (NUM_SEQ, MIN_SEQ_LENGTH)
        self.memmap_dtype = [('url', np.int32, 512), ('tokens', np.int32, (NUM_SEQ, MIN_SEQ_LENGTH)),
                             ('label', np.int16, len(spam_subtype_mapping) - 3),
                             ('pipelines', np.int16, len(spam_subtype_mapping))]

        if os.path.exists(self.memmap_positive_path) and os.path.exists(self.memmap_negative_path):
            print("Loading memmap from disk")
            self.memmap_positive = np.memmap(self.memmap_positive_path, dtype=self.memmap_dtype, mode='r')
            self.memmap_negative = np.memmap(self.memmap_negative_path, dtype=self.memmap_dtype, mode='r')

        # generate memmap if it doesn't exist
        if not os.path.exists(self.memmap_positive_path) or not os.path.exists(self.memmap_negative_path):
            self.memmap_positive = self._load_memmap(self.positive_file, self.memmap_positive_path, is_positive=True)
            self.memmap_negative = self._load_memmap(self.negative_file, self.memmap_negative_path, is_positive=False)

            # save pipeline counts to disk
            with open(
                    data_dir + f"{self.positive_file.split('/')[-1]}_{self.negative_file.split('/')[-1]}_pipeline_counts.pkl",
                    'wb') as f:
                pickle.dump(self.pipeline_counts, f)

        self.samples = []
        for i in range(len(self.memmap_positive)):
            self.samples.append((True, i))
        for i in range(len(self.memmap_negative)):
            self.samples.append((False, i))

        # shuffle the samples
        if shuffle:
            random.shuffle(self.samples)

    def _load_memmap(self, file_path, memmap_path, is_positive):
        dtype = self.memmap_dtype

        # check if memmap exists in data_dir.
        if os.path.exists(memmap_path):
            arr = np.memmap(memmap_path, dtype=dtype, mode='r')
            return arr

        # otherwise, create it.File is a tsv. If positive, it has 5 columns: url, tokens, html, pipeline, label.
        # If negative, it 4 columns: url, tokens, html, label.
        # number of lines in file is number of samples in dataset.
        # create memmap with shape (num_samples,).
        shape = (sum(1 for _ in open(file_path)),)
        mode = 'w+'
        memmap_arr = np.memmap(memmap_path, dtype=dtype, shape=shape, mode=mode)

        # write samples to memmap
        for idx, sample in enumerate(open(file_path, 'rb')):

            sample = sample.decode('utf-8').strip().split('\t')
            try:
                url, tokens, html, pipelines, label = sample
            except:
                url, tokens, html, label = sample
                pipelines = "Negative"

            url_tokens = torch.tensor(
                [self.tokenizer.cls_token_id] + self.tokenizer.encode(url, add_special_tokens=False, max_length=511,
                                                                      padding="max_length", truncation=True))

            label, pipelines = self._generate_label_and_pipeline(pipelines, is_positive)

            memmap_arr[idx]['url'] = url_tokens
            memmap_arr[idx]['tokens'] = self._generate_sample(tokens)
            memmap_arr[idx]['label'] = label
            memmap_arr[idx]['pipelines'] = pipelines

            if idx % 10000 == 0:
                print(f'Processed {idx} samples')

        # write memmap to disk
        memmap_arr.flush()

        return memmap_arr

    def __getitem__(self, index):
        is_positive, index = self.samples[index]
        if is_positive:
            url_tokens, tokens, label, pipelines = self.memmap_positive[index]
        else:
            url_tokens, tokens, label, pipelines = self.memmap_negative[index]

        if sum(pipelines) == 0:
            print("Label: ", label)
            print("Pipelines: ", pipelines)
        return torch.tensor(url_tokens), torch.tensor(tokens), torch.tensor(label, dtype=torch.float32), torch.tensor(
            pipelines, dtype=torch.float32)

    def _generate_sample(self, tokens):
        tokens = [int(t) for t in tokens.split('!')]
        tokens = np.array(tokens).reshape(NUM_SEQ, MIN_SEQ_LENGTH)
        return torch.from_numpy(tokens)

    def _generate_label_and_pipeline(self, pipeline, is_positive):
        label = np.zeros(
            len(spam_subtype_mapping) - 3)  # -3 because we don't care about Empty, Other, or Negative when predicting.
        pipelines = np.zeros(len(spam_subtype_mapping))

        if is_positive:
            label[spam_subtype_reverse_mapping['IsSpam']] = 1

        # strip whitespace and get list of pipelines
        pipeline = [p.strip() for p in pipeline.split(',')]

        # iterate through pipelines and set label
        for p in pipeline:
            if p not in spam_subtype_reverse_mapping:
                continue
            self.pipeline_counts[p] += 1
            if p != 'Empty' and p != 'Other' and p != 'Negative':
                try:
                    label[spam_subtype_reverse_mapping[p]] = 1
                except KeyError:
                    print(f'Pipeline {p} not found in mapping')
                    continue
            pipelines[spam_subtype_reverse_mapping[p]] = 1

        return label, pipelines

    def __len__(self):
        return len(self.samples)


# Define the desired number of samples for each subtype
desired_samples = {
    "BrandStuffing": 500000,
    "KeywordStuffing": 255210,
    "Media": 172854,
    "MGC": 203636,
    "Other": 550000,
    "Empty": 550000,
    "Negative": 2200000,
}

from torch.utils.data.distributed import DistributedSampler


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


class DistributedWeightedSampler(Sampler):
    """
    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same across
        all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
            self,
            weights: list,
            num_samples: int = None,
            replacement: bool = True,
            num_replicas: int = None,
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = int(num_replicas)
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """
        rank = int(os.environ.get("LOCAL_RANK", 0))

        if rank >= self.num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(rank, self.num_replicas - 1)
            )

        weights = self.weights.copy()
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[rank: self.total_num_samples: self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[rank: self.total_num_samples: self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, embedding_dim):
        super(PositionalEncoding, self).__init__()

        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # create the positional encoding table
        pe = torch.zeros(seq_length, embedding_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register the positional encoding table as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add the positional encoding to the input tensor
        x = x + self.pe[:x.size(1), :]
        return x


class SpamModelV2(nn.Module):
    def __init__(self, enc_num_layers=1, combine_num_layers=1):
        super(SpamModelV2, self).__init__()

        self.enc = BertModel.from_pretrained("bert-base-multilingual-cased")
        # keep only first n encoder layer
        self.enc.encoder.layer = self.enc.encoder.layer[:enc_num_layers]

        self.url_enc = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.url_enc.encoder.layer = self.url_enc.encoder.layer[:1]

        self.combine = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.combine.encoder.layer = self.combine.encoder.layer[:combine_num_layers]
        # set to empty so dont waste space.
        self.combine.embeddings.word_embeddings = None
        self.combine.embeddings.position_embeddings = PositionalEncoding(NUM_SEQ + 1, 768)

        self.combine.apply(self._init_weights)

        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(spam_subtype_mapping) - 3),  # empty, negative, other are not included
        )

        self.fc.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, url_x, x):
        # get url embedding
        url_x = self.url_enc(input_ids=url_x, attention_mask=url_x != 0)["pooler_output"]

        # get doc embedding
        x = torch.stack([self.enc(input_ids=x[:, i, :], attention_mask=x[:, i, :] != 0)["pooler_output"] for i in
                         range(x.shape[1])], dim=1)

        # concat url and doc embedding
        x = torch.cat([url_x.unsqueeze(1), x], dim=1)

        # add combiner position embeddings before passing to combiner.
        x = x + self.combine.embeddings.position_embeddings(x)

        # pass through LayerNorm. This will normalize across the embedding dimension.
        x = self.combine.embeddings.LayerNorm(x)

        # dropout
        x = self.combine.embeddings.dropout(x)
        x = self.combine.encoder(x)["last_hidden_state"]
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        print("decay: %s" % (str(decay),))
        print("no_decay: %s" % (str(no_decay),))

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


model = SpamModelV2(enc_num_layers=1, combine_num_layers=1)


######################
# load checkpoint that has keys 'model', 'optimizer' etc. We will load the model weights from this checkpoint.
# checkpoint_path = "/home/bling/checkpoints/ckpt_45000_0.17615531831979753_.pt"
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model'])
# print("model loaded from checkpoint path: ", checkpoint_path)
######################

class CustomMaskedLoss(nn.Module):
    def __init__(self):
        super(CustomMaskedLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, label, pipelines):
        # create mask for relevant labels
        mask = torch.ones_like(label)

        # if pipeline is empty, then mask out all label except for "IsSpam"
        for idx, pipeline in enumerate(pipelines):
            # get pipeline type. Position of "empty", rest will be 0.
            if sum(pipeline) == 1 and pipeline[spam_subtype_reverse_mapping["Empty"]] == 1:
                assert label[idx, spam_subtype_reverse_mapping["IsSpam"]] == 1
                mask[idx, :spam_subtype_reverse_mapping["IsSpam"]] = 0
                mask[idx, spam_subtype_reverse_mapping["IsSpam"]] = 1

        # calculate loss and apply mask
        loss = self.loss(output * mask, label)
        return loss


out_dir = '/scratch/workspaceblobstore/spam/logs/logs_' + str(datetime.datetime.now()).replace(' ', '_').replace(':',
                                                                                                                 '_').replace(
    '.', '_')

tensorboard_dir = sys.argv[sys.argv.index('--tensorboard_dir') + 1]
print("writing tensorboard logs to dir : ", tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)

# adamw optimizer
learning_rate = 1e-4  # max learning rate
max_iters = 58000  # total number of training iterations
weight_decay = 1e-1
betas = (0.9, 0.95)
grad_clip = 1.0  # clip gradients at this value
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 4000  # how many steps to warm up for
lr_decay_iters = 58000  # how many steps to decay the learning rate for
min_lr = 1e-5  # minimum learning rate

# data related
dtype = 'float16'
gradient_accumulation_steps = 4  # how many batches to accumulate gradients for
eval_interval = 200  # how often to evaluate the model on the validation set
log_interval = 10  # how often to log training information

# model settings
device = 'cuda:0'

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1  # is this a ddp run?
print("ddp: ", ddp)

if ddp:
    init_process_group(backend=backend)
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
else:
    gpu_id = 0  # gpu_id 0 means this is the (single) master process, basically

if gpu_id == 0:
    os.makedirs(out_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

model = model.to(device)

# initialize a GradScaler if data type is float16 else no-op.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# note: float16 data type will automatically use a GradScaler
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
from contextlib import nullcontext

ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas)

# print("compiling model.....")
# compile the model
# model = torch.compile(model) # requires pytorch 2.0
# print("done compiling model.")

if ddp:
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    print("model wrapped in DDP...")


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_metrics(preds_list, label_list):
    accuracy = sum(1 for x in range(len(label_list)) if preds_list[x] == label_list[x]) / len(label_list)
    precision, recall, _, _ = precision_recall_fscore_support(label_list, preds_list, average=None)
    fpr = sum(1 for x in range(len(label_list)) if preds_list[x] == 1 and label_list[x] == 0) / sum(
        1 for x in range(len(label_list)) if label_list[x] == 0)
    return accuracy, precision[1], recall[1], fpr


def eval_binary_classifier(model, dataloader, iter_num) -> None:
    print("evaluating...")
    model.eval()  # set model to evaluation mode

    label_list = []
    output_list = []
    losses = []

    loss_function = CustomMaskedLoss()

    # iterate over the data
    for idx, (url_tokens, inputs, labels, pipelines) in enumerate(dataloader):
        # move data to the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)
        url_tokens = url_tokens.to(device)

        # get model outputs
        with torch.no_grad():
            outputs = model(url_tokens, inputs)
            loss = loss_function(outputs, labels, pipelines)

        output_list.extend(outputs.view(-1).tolist())
        label_list.extend(labels.view(-1).tolist())
        losses.append(loss.item())

    # get all negatives
    # Each label was of length 5. After extend to label list, Every 5th element contains IsSpam. Get all elements where IsSpam is 0.
    neg_indices = []
    for idx in range(spam_subtype_reverse_mapping['IsSpam'], len(label_list), 5):
        if label_list[idx] == 0:
            neg_indices.append(idx)

    print("num negatives: ", len(neg_indices))

    # construct PR curve for each type of label. See spam_subtype_mapping for each type of label.
    for idx in range(len(spam_subtype_mapping) - 3):  # -3 because we don't care about Other, Empty, Negative.
        indices = []
        # get the indices of the current label from label_list. Every 5th element is the current label.
        for i in range(idx, len(label_list), 5):
            if label_list[i] == 1:
                indices.append(i)

        # get the corresponding outputs and labels.
        output_list_sub = [output_list[i] for i in indices]
        label_list_sub = [label_list[i] for i in indices]

        if len(output_list_sub) == 0:
            continue

        # add an equal number of negatives to the current label. Sample randomly from the negatives.
        if neg_indices:
            neg_indices_sub = random.sample(neg_indices, len(output_list_sub))
            output_list_sub.extend([output_list[i] for i in neg_indices_sub])
            label_list_sub.extend([label_list[i] for i in neg_indices_sub])

        print("label_list for label {}: {}".format(spam_subtype_mapping[idx], label_list_sub[:100]))
        print("num of 1, num of 0 in label_list for label {}: {}, {}".format(spam_subtype_mapping[idx],
                                                                             sum(label_list_sub),
                                                                             len(label_list_sub) - sum(label_list_sub)))

        pr, re, thr = precision_recall_curve(label_list_sub, output_list_sub)
        pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
        auprc = auc(pr_curve[0], pr_curve[1])
        print("auprc for label {}: {}".format(spam_subtype_mapping[idx], auprc))

        # write to tensorboard
        writer.add_pr_curve("val/AUPRC_{}".format(spam_subtype_mapping[idx]), np.array(label_list_sub),
                            np.array(output_list_sub), global_step=iter_num)

    # pr, re, thr = precision_recall_curve(label_list, output_list)
    # pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
    # auprc = auc(pr_curve[0], pr_curve[1])

    # print("Overall auprc: {}".format(auprc))
    # writer.add_pr_curve("val/AUPRC", np.array(label_list), np.array(output_list), global_step=iter_num)
    # writer.add_scalar("val/loss", np.mean(losses), global_step=iter_num)

    # reset to train mode
    model.train()


def train(model, batch_size, data_dir):
    loss_function = CustomMaskedLoss()
    iter_num = 0

    t0 = time.time()

    train_pos_file = os.path.join(data_dir, "positives_sample.tsv")
    valid_pos_file = os.path.join(data_dir, "positives_sample.tsv")
    train_neg_file = os.path.join(data_dir, "negatives_sample.tsv")
    valid_neg_file = os.path.join(data_dir, "negatives_sample.tsv")

    # load the data
    train_dataset = SpamDataset(train_pos_file, train_neg_file)
    valid_dataset = SpamDataset(valid_pos_file, valid_neg_file)

    num_samples_seen_per_pipeline = defaultdict(int)

    # Calculate weights for each sample based on their subtype
    # check if weight can be loaded from saved file
    if os.path.exists(data_dir + 'weights.pkl'):
        with open(data_dir + 'weights.pkl', 'rb') as f:
            weights = pickle.load(f)
    else:
        weights = []
        # Calculate the total desired samples per epoch
        for _, _, _, pipeline in train_dataset:
            # get subtype from mapping. If all zeros, then it is "Negative".
            subtypes = [spam_subtype_mapping[i] for i, v in enumerate(pipeline) if v == 1]
            if len(subtypes) == 0:
                subtypes = ["Negative"]
            weight = sum([desired_samples[subtype] / train_dataset.pipeline_counts[subtype] for subtype in subtypes])
            weights.append(weight)

        # save weights to file
        with open(data_dir + 'weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

    print(weights)
    # Create a WeightedRandomSampler instance
    sampler_ = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    sampler = DistributedProxySampler(sampler_, num_replicas=4, rank=gpu_id)
    # TODO: Check if this is correct.
    # sampler = DistributedWeightedSampler(weights=weights, num_samples=len(train_dataset), replacement=True)

    # Create a DataLoader with the created sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    # train_loader.sampler.set_epoch(0)

    # Create a DataLoader without sampler for validation
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    print("Loaded {} training samples".format(len(train_dataset)))

    while True:

        for batch in train_loader:

            # determine the learning rate for this iteration
            if decay_lr:
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = learning_rate

            # evaluate on val if gpu_id == 0
            if gpu_id == 0 and iter_num % eval_interval == 0:
                eval_binary_classifier(model, valid_loader, iter_num)
                raw_model = model.module if hasattr(model, "module") else model

                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'lr': 'lr'
                }

                checkpoint_name = f"checkpoint_iter_{iter_num}"
                print("saving checkpoint to path: ", checkpoint_name)
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))

                # reset to train mode
                model.train()

                # train
            url_tokens, data, labels, pipelines = batch

            # in pipelines, count number of 1s across all samples for each subtype.
            if gpu_id == 0:
                total = 0
                for idx in range(pipelines.shape[1]):
                    if idx != spam_subtype_reverse_mapping["IsSpam"]:
                        num_samples_seen_per_pipeline[idx] += pipelines[:, idx].sum().item()
                        total += num_samples_seen_per_pipeline[idx]

                # if pipeline is "IsSpam", then its value is sum of all other pipelines plus "Negative".
                num_samples_seen_per_pipeline[spam_subtype_reverse_mapping["IsSpam"]] = total - \
                                                                                        num_samples_seen_per_pipeline[
                                                                                            spam_subtype_reverse_mapping[
                                                                                                "Negative"]]

            if ddp:
                model.require_backward_grad_sync = (iter_num % gradient_accumulation_steps == 0)

            with ctx:
                scores = model(url_tokens.to(device), data.to(device)).squeeze(1)

            labels = labels.to(device)

            loss = loss_function(scores, labels.float(), pipelines)

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

            if iter_num % gradient_accumulation_steps == 0:
                # clip gradients
                if grad_clip != 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # step the optimizer
                scaler.step(optimizer)
                scaler.update()

                # flush the gradients
                optimizer.zero_grad(set_to_none=False)

            # log training loss
            if gpu_id == 0 and iter_num % log_interval == 0:
                lossf = loss.item()
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

                # log to tensorboard
                writer.add_scalar('train/loss', loss.item(), iter_num)
                writer.add_scalar('train/lr', lr, iter_num)

                try:
                    # get number of gpus in local node
                    num_gpus = torch.cuda.device_count()
                    num_examples_seen = iter_num * batch_size * num_gpus
                    writer.add_scalar('train/num_examples_seen', num_examples_seen, iter_num)
                except:
                    print("Error logging weights to tensorboard for iter_num: ", iter_num)

                # log number of samples seen per pipeline
                for idx, num_samples in num_samples_seen_per_pipeline.items():
                    writer.add_scalar(f'train/num_samples_seen_per_pipeline/{idx}', num_samples, iter_num)
                    print(f"iter {iter_num}: num_samples_seen for {spam_subtype_mapping[idx]}: {num_samples}")

            iter_num += 1
            if iter_num >= max_iters:
                return


if __name__ == "__main__":
    batch_size = 128
    train(model, batch_size, data_dir)

