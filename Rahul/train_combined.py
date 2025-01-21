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
import subprocess
from torch.utils.tensorboard import SummaryWriter

# data directory
data_dir = '/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/'
tensorboard_dir = sys.argv[sys.argv.index('--tensorboard_dir') + 1]
print("writing tensorboard logs to dir : ", tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)
out_dir = '/scratch/workspaceblobstore/spam/logs/logs_' + str(tensorboard_dir.split('/')[-1]) + '/'

# check if --finetune is set in the system arguments.
is_bs_finetune = '--bs_finetune' in sys.argv
print("This is a bs finetune run: ", is_bs_finetune)

is_finetune = '--finetune' in sys.argv
is_finetune = is_finetune or is_bs_finetune

print("This is a finetune run: ", is_finetune)
if is_finetune:
    data_dir = data_dir + 'finetune/'

MANUAL_SEED = 42  # meaning of life, the universe, and everything.
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed_all(MANUAL_SEED)  # if you are using multi-GPU.
np.random.seed(MANUAL_SEED)  # Numpy module.
random.seed(MANUAL_SEED)  # Python random module.

# avoid non deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

average_embeddings = sys.argv[sys.argv.index(
    '--average_embeddings') + 1] == 'True' if '--average_embeddings' in sys.argv else False
eval_only = '--eval_only' in sys.argv
eval_oct_leakage = '--eval_oct_leakage' in sys.argv
eval_april_leakage = '--eval_april_leakage' in sys.argv
eval_mikhail_set = '--eval_mikhail_set' in sys.argv

finetune_dropout_replace_experiment = '--finetune_dropout_replace_experiment' in sys.argv

# Define the desired number of samples for each subtype
if is_finetune:
    # get num_negatives from system arguments if specified, else set to 9500000.
    num_negatives = int(sys.argv[sys.argv.index('--num_negatives') + 1]) if '--num_negatives' in sys.argv else 9500000

    desired_samples = {
        "BrandStuffing": 1003128,
        "KeywordStuffing": 284661,
        "Media": 88758,
        "MGC": 480006,  # till here, sums to 1856553
        "Other": 1142656,  # till here, everything is pretty much exact counts.
        "Empty": 54424,  # oversampling this by 8x.  total so far: 3053633
        "Negative": num_negatives,
    }

else:
    desired_samples = {
        "BrandStuffing": 631700,
        "KeywordStuffing": 255210,
        "Media": 172854,
        "MGC": 203636,
        "Other": 631700,
        "Empty": 631700,
        "Negative": 3000000,
    }

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
    def __init__(self, positive_file, negative_file, positive_count=0, negative_count=0, shuffle=True, is_eval=False):
        # Store the paths to the positive and negative files and compute offsets
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.tokenizer = tokenizer
        self.is_eval = is_eval

        self.pipeline_counts = defaultdict(int)
        # if pipeline_counts exists, load it from disk.
        if os.path.exists(
                data_dir + f"{self.positive_file.split('/')[-1]}_{self.negative_file.split('/')[-1]}_pipeline_counts.pkl"):
            with open(
                    data_dir + f"{self.positive_file.split('/')[-1]}_{self.negative_file.split('/')[-1]}_pipeline_counts.pkl",
                    'rb') as f:
                self.pipeline_counts = pickle.load(f)
            print("Loaded pipeline counts from disk")

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
            print("Loaded memmap from disk")

        # generate memmap if it doesn't exist
        if not os.path.exists(self.memmap_positive_path) or not os.path.exists(self.memmap_negative_path):
            self.memmap_positive = self._load_memmap(self.positive_file, self.memmap_positive_path,
                                                     count=positive_count)
            self.memmap_negative = self._load_memmap(self.negative_file, self.memmap_negative_path,
                                                     count=negative_count)

            # save pipeline counts to disk
            with open(
                    data_dir + f"{self.positive_file.split('/')[-1]}_{self.negative_file.split('/')[-1]}_pipeline_counts.pkl",
                    'wb') as f:
                pickle.dump(self.pipeline_counts, f)

        print("Creating samples..")
        self.samples = []
        for i in range(len(self.memmap_positive)):
            self.samples.append((True, i))
        for i in range(len(self.memmap_negative)):
            self.samples.append((False, i))
        print("Done creating samples")

        # shuffle the samples
        if shuffle:
            random.shuffle(self.samples)

        print("Done shuffling samples")

    def _load_memmap(self, file_path, memmap_path, count):
        dtype = self.memmap_dtype

        # check if memmap exists in data_dir.
        if os.path.exists(memmap_path):
            arr = np.memmap(memmap_path, dtype=dtype, mode='r')
            return arr

        # otherwise, create it.File is a tsv. If positive, it has 5 columns: url, tokens, html, pipeline, label.
        # If negative, it 4 columns: url, tokens, html, label.
        # number of lines in file is number of samples in dataset.
        # create memmap with shape (num_samples,).
        print("calculating shape of memmap...")
        if count == 0:
            def wccount(filename):
                out = subprocess.check_output(['wc', '-l', filename])
                return int(out.decode().split()[0])

            shape = (wccount(file_path),)
        else:
            shape = (count,)

        print("shape of memmap: ", shape)
        mode = 'w+'
        memmap_arr = np.memmap(memmap_path, dtype=dtype, shape=shape, mode=mode)

        # write samples to memmap. Open only in 'rb' mode, file is too large to open in 'r' mode.
        print("Starting to read file: ", file_path)
        for idx, sample in enumerate(open(file_path, 'rb')):

            try:
                sample = sample.decode('utf-8').strip().split('\t')
                try:
                    url, tokens, html, pipelines, label = sample
                except:
                    url, tokens, html, label = sample
                    pipelines = "Negative"

                # if pipeline is negative, assert that label is 0.
                if pipelines == "Negative":
                    assert label == '0', f"Label is {label} for negative sample"

                url_tokens = torch.tensor(
                    [self.tokenizer.cls_token_id] + self.tokenizer.encode(url, add_special_tokens=False, max_length=511,
                                                                          padding="max_length", truncation=True))

                label, pipelines = self._generate_label_and_pipeline(pipelines)

                memmap_arr[idx]['url'] = url_tokens
                memmap_arr[idx]['tokens'] = self._generate_sample(tokens)
                memmap_arr[idx]['label'] = label
                memmap_arr[idx]['pipelines'] = pipelines

                if idx % 10000 == 0:
                    print(f'Processed {idx} samples')

                if count != 0 and idx == count - 1:
                    break

            except Exception as e:
                print("Skipping sample at index: ", idx)
                print(e)

        # write memmap to disk
        memmap_arr.flush()

        return memmap_arr

    def __getitem__(self, index):
        is_positive_file, index = self.samples[index]
        if is_positive_file:
            url_tokens, tokens, label, pipelines = self.memmap_positive[index]
        else:
            url_tokens, tokens, label, pipelines = self.memmap_negative[index]

        # for a sample test, replace url_tokens with a random tensor. Should be shape (512,), start with 101. Max value is 110000.
        # generate random number between 0 and 1. If < 0.5, replace with random url from positive dataset. Otherwise, replace with random url from negative dataset.

        if finetune_dropout_replace_experiment:
            # if dropout percentage specified in system arguments, use that. Otherwise, use 0.3.
            dropout_percentage = float(sys.argv[sys.argv.index('--dropout') + 1]) if '--dropout' in sys.argv else 0.3
            random_num = random.random()
            # url experiment
            try:
                if random_num > 0.8 and random_num <= 0.9:
                    url_tokens = self.memmap_positive[random.randint(0, len(self.memmap_positive))]['url']
                elif random_num > 0.9:
                    url_tokens = self.memmap_negative[random.randint(0, len(self.memmap_negative))]['url']
            except:
                pass

            # tokens experiment. replace tokens with zero tensor when p between 0.5 and 0.8.
            try:
                random_number_floor = 0.8 - dropout_percentage
                if random_num > random_number_floor and random_num <= 0.8:
                    tokens = torch.zeros_like(tokens)
            except:
                pass

        return torch.tensor(url_tokens), torch.tensor(tokens), torch.tensor(label, dtype=torch.float32), torch.tensor(
            pipelines, dtype=torch.float32)

    def _generate_sample(self, tokens):
        tokens = [int(t) for t in tokens.split('!')]
        tokens = np.array(tokens).reshape(NUM_SEQ, MIN_SEQ_LENGTH)
        return torch.from_numpy(tokens)

    def _generate_label_and_pipeline(self, pipeline):
        label = np.zeros(
            len(spam_subtype_mapping) - 3)  # -3 because we don't care about Empty, Other, or Negative when predicting.
        pipelines = np.zeros(len(spam_subtype_mapping))

        if "Negative" in pipeline:
            # split pipeline by comma, assert that "Negative" is the only pipeline.
            assert len(pipeline.split(',')) == 1, f"Pipeline is {pipeline}, should only be Negative"
            label[spam_subtype_reverse_mapping['IsSpam']] = 0
        else:
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


class SpamMonoDataset(Dataset):
    def __init__(self, positive_file, positive_count=0, shuffle=True, train=False):
        # Store the paths to the positive and negative files and compute offsets
        self.positive_file = positive_file
        self.tokenizer = tokenizer
        self.train = train

        self.pipeline_counts = defaultdict(int)
        # if pipeline_counts exists, load it from disk.
        if os.path.exists(data_dir + f"{self.positive_file.split('/')[-1]}_pipeline_counts.pkl"):
            with open(data_dir + f"{self.positive_file.split('/')[-1]}_pipeline_counts.pkl", 'rb') as f:
                self.pipeline_counts = pickle.load(f)
            print("Loaded pipeline counts from disk: ", self.pipeline_counts)

        self.memmap_positive_path = self.positive_file + '.memmap'
        # tokens is of shape (NUM_SEQ, MIN_SEQ_LENGTH)
        self.memmap_dtype = [('url', np.int32, 512), ('tokens', np.int32, (NUM_SEQ, MIN_SEQ_LENGTH)),
                             ('label', np.int16, len(spam_subtype_mapping) - 3),
                             ('pipelines', np.int16, len(spam_subtype_mapping))]

        if os.path.exists(self.memmap_positive_path):
            print("Loading memmap from disk")
            self.memmap_positive = np.memmap(self.memmap_positive_path, dtype=self.memmap_dtype, mode='r')
            print("Loaded memmap from disk")

        # generate memmap if it doesn't exist
        if not os.path.exists(self.memmap_positive_path):
            self.memmap_positive = self._load_memmap(self.positive_file, self.memmap_positive_path,
                                                     count=positive_count)

            # save pipeline counts to disk
            with open(data_dir + f"{self.positive_file.split('/')[-1]}_pipeline_counts.pkl", 'wb') as f:
                pickle.dump(self.pipeline_counts, f)
            print("Saved pipeline counts to disk: ", self.pipeline_counts)

        print("Creating samples..")
        self.samples = []
        for i in range(len(self.memmap_positive)):
            self.samples.append((True, i))
        print("Done creating samples")

        # shuffle the samples
        if shuffle:
            random.shuffle(self.samples)

        print("Done shuffling samples")

    def _load_memmap(self, file_path, memmap_path, count):
        dtype = self.memmap_dtype

        # check if memmap exists in data_dir.
        if os.path.exists(memmap_path):
            arr = np.memmap(memmap_path, dtype=dtype, mode='r')
            return arr

        # otherwise, create it.File is a tsv. If positive, it has 5 columns: url, tokens, html, pipeline, label.
        # If negative, it 4 columns: url, tokens, html, label.
        # number of lines in file is number of samples in dataset.
        # create memmap with shape (num_samples,).
        print("calculating shape of memmap...")
        if count == 0:
            def wccount(filename):
                out = subprocess.Popen(['wc', '-l', filename],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT
                                       ).communicate()[0]
                return int(out.partition(b' ')[0])

            shape = (wccount(file_path),)
        else:
            shape = (count,)

        print("shape of memmap: ", shape)
        mode = 'w+'
        memmap_arr = np.memmap(memmap_path, dtype=dtype, shape=shape, mode=mode)

        # write samples to memmap. Open only in 'rb' mode, file is too large to open in 'r' mode.
        print("Starting to read file: ", file_path)
        for idx, sample in enumerate(open(file_path, 'rb')):

            sample = sample.decode('utf-8').strip().split('\t')
            url, pipelines, tokens, label, html = None, None, None, None, None

            if eval_oct_leakage:
                label, url, tokens = sample
                if float(label) == 0.0:
                    pipelines = "Negative"
                elif float(label) == 1:
                    pipelines = "Empty"

            elif eval_april_leakage:
                url, tokens, html, label = sample
                # convert label from False/True to 0/1
                label = 1 if label == 'True' else 0
                pipelines = "Negative" if label == 0 else "Empty"

            else:  # used to be eval_mikhail_set
                url, tokens, html, pipelines, label = sample

            # if pipeline is negative, assert that label is 0.
            if pipelines == "Negative":
                assert float(label) == 0, f"Label is {label} for negative sample"

            assert url is not None and tokens is not None and pipelines is not None and label is not None, "One of url, tokens, pipelines, or label is None"

            url_tokens = torch.tensor(
                [self.tokenizer.cls_token_id] + self.tokenizer.encode(url, add_special_tokens=False, max_length=511,
                                                                      padding="max_length", truncation=True))

            label, pipelines = self._generate_label_and_pipeline(pipelines)

            memmap_arr[idx]['url'] = url_tokens
            memmap_arr[idx]['tokens'] = self._generate_sample(tokens)
            memmap_arr[idx]['label'] = label
            memmap_arr[idx]['pipelines'] = pipelines

            if idx % 10000 == 0:
                print(f'Processed {idx} samples')

            if count != 0 and idx == count - 1:
                break

        # write memmap to disk
        memmap_arr.flush()

        return memmap_arr

    def __getitem__(self, index):
        url_tokens, tokens, label, pipelines = self.memmap_positive[index]
        if self.train and finetune_dropout_replace_experiment:
            # if dropout percentage specified in system arguments, use that. Otherwise, use 0.3.
            dropout_percentage = float(sys.argv[sys.argv.index('--dropout') + 1]) if '--dropout' in sys.argv else 0.3
            random_num = random.random()
            # url experiment
            try:
                if random_num > 0.8 and random_num <= 0.9:
                    url_tokens = self.memmap_positive[random.randint(0, len(self.memmap_positive))]['url']
                elif random_num > 0.9:
                    url_tokens = self.memmap_negative[random.randint(0, len(self.memmap_negative))]['url']
            except:
                pass

            # tokens experiment. replace tokens with zero tensor when p between 0.5 and 0.8.
            try:
                random_number_floor = 0.8 - dropout_percentage
                if random_num > random_number_floor and random_num <= 0.8:
                    tokens = torch.zeros_like(tokens)
            except:
                pass

        return torch.tensor(url_tokens), torch.tensor(tokens), torch.tensor(label, dtype=torch.float32), torch.tensor(
            pipelines, dtype=torch.float32)

    def _generate_sample(self, tokens):
        tokens = [int(t) for t in tokens.split('!')]
        tokens = np.array(tokens).reshape(NUM_SEQ, MIN_SEQ_LENGTH)
        return torch.from_numpy(tokens)

    def _generate_label_and_pipeline(self, pipeline):
        label = np.zeros(
            len(spam_subtype_mapping) - 3)  # -3 because we don't care about Empty, Other, or Negative when predicting.
        pipelines = np.zeros(len(spam_subtype_mapping))

        # strip whitespace and get list of pipelines
        pipeline = [p.strip() for p in pipeline.split(',')]

        # if "Negative" is the only pipeline, set label to 0. Else, set label to 1.
        # If negative in pipeline, assert that it is the only pipeline.
        if "Negative" in pipeline:
            assert len(pipeline) == 1, f"Pipeline is {pipeline} for negative sample"
            label[spam_subtype_reverse_mapping['IsSpam']] = 0
        else:
            label[spam_subtype_reverse_mapping['IsSpam']] = 1

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

        if is_bs_finetune:
            # everything so far is frozen.
            for param in self.parameters():
                param.requires_grad = False

            # add a 256 x 128 x 1 linear layer from (768, 256) in self.fc
            self.fc_bs = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            self.fc_bs.apply(self._init_weights)

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

        # average across the sequence dimension
        x = torch.mean(x, dim=1) if average_embeddings else x[:, 0, :]

        if not is_bs_finetune:
            x = self.fc(x)

        else:
            # pass through first two layers of fc
            x = self.fc[:2](x)
            # pass through fc_bs
            x = self.fc_bs(x)

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


# get enc_num_layers from command line. If not provided, default to 1.
enc_num_layers = sys.argv[sys.argv.index("--enc_num_layers") + 1] if "--enc_num_layers" in sys.argv else 1
model = SpamModelV2(enc_num_layers=int(enc_num_layers), combine_num_layers=1)

if is_bs_finetune:
    # assert that all layers except self.fc[2] are frozen.
    # for name, param in model.named_parameters():
    #     if name.startswith("fc.2"):
    #         assert param.requires_grad == True
    #     else:
    #         assert param.requires_grad == False

    # assert that all layers except self.fc_bs are frozen.
    for name, param in model.named_parameters():
        if name.startswith("fc_bs"):
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False

# adamw optimizer
max_iters = 60000  # total number of training iterations
grad_clip = 1.0  # clip gradients at this value
# learning rate decay settings
decay_lr = True if not is_bs_finetune else False  # whether to decay the learning rate
warmup_iters = 4000  # how many steps to warm up for
lr_decay_iters = 80000  # how many steps to decay the learning rate for
weight_decay = 1e-1
betas = (0.9, 0.95)
learning_rate = 1e-5  # max learning rate
min_lr = 1e-6  # minimum learning rate

optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas)


class CustomMaskedLoss(nn.Module):
    def __init__(self):
        super(CustomMaskedLoss, self).__init__()
        weight = (521 + 1400000) / (2 * 521)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight)) if is_bs_finetune else nn.BCEWithLogitsLoss()

    def forward(self, output, label, pipelines):
        # create mask for relevant labels
        mask = torch.ones_like(label)

        # if pipeline is empty, then mask out all label except for "IsSpam"
        for idx, pipeline in enumerate(pipelines):
            if is_bs_finetune:
                # set brand stuffing to 1, rest to 0.
                mask[idx, 0:] = 0
                mask[idx, 0] = 1

            else:
                # get pipeline type. Position of "empty", rest will be 0.
                if sum(pipeline) == 1 and pipeline[spam_subtype_reverse_mapping["Empty"]] == 1:
                    assert label[idx, spam_subtype_reverse_mapping["IsSpam"]] == 1
                    mask[idx, :spam_subtype_reverse_mapping["IsSpam"]] = 0
                    mask[idx, spam_subtype_reverse_mapping["IsSpam"]] = 1

        # calculate loss and apply mask
        loss = self.loss(output, label)
        loss = loss * mask

        return loss.mean()


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


# data related
dtype = 'float16'
gradient_accumulation_steps = 4  # how many batches to accumulate gradients for
eval_interval = 4000 if not is_bs_finetune else 12000  # how often to evaluate the model on the validation set
log_interval = 250  # how often to log training information

# model settings
device = 'cuda:0'

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1  # is this a ddp run?
print("ddp: ", ddp)

world_size = torch.cuda.device_count() if ddp else 1  # how many processes are there in total

if ddp:
    from datetime import timedelta

    timeout = timedelta(hours=5)
    init_process_group(backend=backend, timeout=timeout)
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

######################
# if resume training, load provided checkpoint.
prev_iter_num = 0
prev_num_samples_seen_per_pipeline = None
if "--resume_from_checkpoint" in sys.argv:
    checkpoint_path = sys.argv[sys.argv.index("--resume_from_checkpoint") + 1]
    checkpoint = torch.load(checkpoint_path)

    # if keys start from _orig_mod., then remove _orig_mod. from keys
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[10:]] = checkpoint['model'][key]
            del checkpoint['model'][key]

    # Create a copy of checkpoint model state
    new_state_dict = checkpoint['model'].copy()

    # model.load_state_dict(new_state_dict) if not is_bs_finetune else model.load_state_dict(new_state_dict, strict=False)
    model.load_state_dict(new_state_dict, strict=False)
    print("model loaded from checkpoint path: ", checkpoint_path)

    # load optimizer state
    if not is_bs_finetune and not eval_april_leakage:
        optimizer.load_state_dict(checkpoint['optimizer'])


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
    # load prev_iter_num
    prev_iter_num = int(checkpoint['iter_num'])
    print("prev_iter_num loaded: ", prev_iter_num)

    # load prev_num_samples_seen_per_pipeline
    if 'num_samples_seen_per_pipeline' in checkpoint:
        prev_num_samples_seen_per_pipeline = checkpoint['num_samples_seen_per_pipeline']
        print("prev_num_samples_seen_per_pipeline loaded: ", prev_num_samples_seen_per_pipeline)

# load checkpoint that has keys 'model', 'optimizer' etc. We will load the model weights from this checkpoint.
# checkpoint_path = "/home/bling/checkpoints/ckpt_45000_0.17615531831979753_.pt"
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model'])
# print("model loaded from checkpoint path: ", checkpoint_path)
######################

# compile the model if torch 2.0 is available
if torch.__version__.startswith("2."):
    print("compiling model.....")
    model = torch.compile(model)  # requires torch 2.0
    print("done compiling model.")

if ddp:
    model = DDP(model, device_ids=[gpu_id], find_unused_parameters=False if is_bs_finetune else True)
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


def decode_url(url_token):
    # decode using huggingface mbert tokenizer
    return tokenizer.decode(url_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def eval_binary_classifier(model, dataloader, iter_num, is_mikhail=False) -> None:
    print("evaluating...")
    model.eval()  # set model to evaluation mode

    if is_mikhail:
        print("evaluating on mikhail set...")

    label_list = []
    output_list = []
    losses = []

    num_positives, num_negatives = 0, 0

    eval_mikhail_set = is_mikhail or "--eval_mikhail_set" in sys.argv

    loss_function = CustomMaskedLoss() if not is_bs_finetune else nn.BCEWithLogitsLoss()

    for idx, (url_tokens, inputs, labels, pipelines) in enumerate(dataloader):
        # move data to the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)
        url_tokens = url_tokens.to(device)

        # get model outputs
        with torch.no_grad():
            outputs = F.sigmoid(model(url_tokens, inputs))

        loss = loss_function(outputs, labels, pipelines) if not is_bs_finetune else loss_function(outputs.squeeze(1),
                                                                                                  labels.float()[:, 0])

        output_list.extend(outputs.view(-1).tolist())
        label_list.extend(labels.view(-1).tolist())

        # update num_positives and num_negatives from labels
        num_positives += int(sum(labels.view(-1).tolist()))
        num_negatives += int(len(labels.view(-1).tolist()) - sum(labels.view(-1).tolist()))

        losses.append(loss.item())

        if not (eval_mikhail_set or eval_oct_leakage or eval_april_leakage or eval_only) and len(
                label_list) >= 400000 * 5 and not is_bs_finetune:
            break

        # on mikhail validation set, there are 523 spam. At 1:1000, we have enough negatives.
        # num_positives_reqd = 1046 if not eval_only else 1214
        # num_positives_reqd = 1214 if ("--eval_mikhail_set" in sys.argv or is_bs_finetune) else 1046
        num_positives_reqd = 1214
        if (
                eval_mikhail_set or is_bs_finetune) and num_positives == num_positives_reqd and num_negatives >= num_positives * 500:
            break

    print("Length of label list: , num samples corresponding to it: ", len(label_list), len(label_list) // 5)
    print("Number of positives: ", num_positives)
    print("Number of negatives: ", num_negatives)

    # construct PR curve for each type of label. See spam_subtype_mapping for each type of label.
    for idx in range(5):
        indices = []
        neg_indices = []
        # get the indices of the current label from label_list. Every 5th element is the current label.
        for i in range(idx, len(label_list), 5):
            if label_list[i] == 1:
                indices.append(i)
            elif label_list[i] == 0:
                neg_indices.append(i)

        # get the corresponding outputs and labels and url
        output_list_sub = [output_list[i] if not is_bs_finetune else output_list[i // 5] for i in indices]
        label_list_sub = [label_list[i] for i in indices]

        if len(output_list_sub) == 0:
            print("output_list_sub is empty for label: ", spam_subtype_mapping[idx])
            continue

        # add an equal number of negatives to the current label. Sample randomly from the negatives.
        if neg_indices:
            if eval_mikhail_set or is_bs_finetune:
                # randomly sample from neg_indices
                neg_indices_sub = random.sample(neg_indices, min(500 * num_positives, len(neg_indices)))
            elif eval_oct_leakage or eval_april_leakage:
                neg_indices_sub = neg_indices
            else:
                neg_indices_sub = random.sample(neg_indices, min(len(output_list_sub) * 4, len(neg_indices))) if not (
                            eval_oct_leakage or eval_mikhail_set or is_bs_finetune) else neg_indices

            output_list_sub.extend(
                [output_list[i] if not is_bs_finetune else output_list[i // 5] for i in neg_indices_sub])
            label_list_sub.extend([label_list[i] for i in neg_indices_sub])

        print("label_list for label {}: {}".format(spam_subtype_mapping[idx], label_list_sub[:100]))
        print("num of 1, num of 0 in label_list for label {}: {}, {}".format(spam_subtype_mapping[idx],
                                                                             sum(label_list_sub),
                                                                             len(label_list_sub) - sum(label_list_sub)))

        pr, re, thr = precision_recall_curve(label_list_sub, output_list_sub)
        # print top 5 precisions after sorting where recall > 0.10.
        sorted_pr = sorted(zip(pr, re, thr), key=lambda x: x[0], reverse=True)
        # get top 5 pr where recall > 0.10
        top_5_pr = [x for x in sorted_pr if x[1] > 0.10][:5]
        print("top 5 pr for label with recall > 0.1 {}: {}".format(spam_subtype_mapping[idx], top_5_pr))

        # print recall at 0.98 precision
        recall_at_98_precision = [x[1] for x in sorted_pr if x[0] > 0.98]
        print("recall at 0.98 precision for label {}: {}".format(spam_subtype_mapping[idx], recall_at_98_precision[
            0] if recall_at_98_precision else 0))

        # print precision at 86% recall
        precision_at_86_recall = [x[0] for x in sorted_pr if x[1] > 0.86]
        print("precision at 86% recall for label {}: {}".format(spam_subtype_mapping[idx], precision_at_86_recall[
            0] if precision_at_86_recall else 0))

        # get fpr at threshold specified in function call
        thresholds_to_consider = [0.9603, 0.9762, 0.9683]
        for threshold in thresholds_to_consider:
            fpr = sum(1 for x in range(len(label_list_sub)) if
                      label_list_sub[x] == 0 and output_list_sub[x] > threshold) / sum(
                1 for x in range(len(label_list_sub)) if label_list_sub[x] == 0)
            print("fpr for label {} at threshold {}: {}".format(spam_subtype_mapping[idx], threshold, fpr))

        # get all points where label is 1 and prediction is 0 for best threshold.
        # these are the false negatives.
        # fn = [label_list_sub[i] for i in range(len(label_list_sub)) if label_list_sub[i] == 1 and output_list_sub[i] < top_5_pr[0][2]]

        # get all points where label is 0 and prediction is 1 for best threshold.
        # these are the false positives.
        # fp = [label_list_sub[i] for i in range(len(label_list_sub)) if label_list_sub[i] == 0 and output_list_sub[i] > top_5_pr[0][2]]

        pr_curve = list(zip(*sorted(zip(re, pr), key=lambda x: x[0])))
        auprc = auc(pr_curve[0], pr_curve[1])
        print("auprc for label {}: {}".format(spam_subtype_mapping[idx], auprc))

        # write to tensorboard
        if not is_mikhail:
            writer.add_pr_curve("val/AUPRC_{}".format(spam_subtype_mapping[idx]), np.array(label_list_sub),
                                np.array(output_list_sub), global_step=iter_num, num_thresholds=1000)
        else:
            writer.add_pr_curve("val/AUPRC_{}_mikhail".format(spam_subtype_mapping[idx]), np.array(label_list_sub),
                                np.array(output_list_sub), global_step=iter_num, num_thresholds=1000)

        if eval_april_leakage:
            # write to tensorboard
            writer.add_pr_curve("val/AUPRC_{}_april_leakage".format(spam_subtype_mapping[idx]),
                                np.array(label_list_sub), np.array(output_list_sub), global_step=iter_num,
                                num_thresholds=1000)

    # reset to train mode
    model.train()


def train(model, batch_size, data_dir):
    loss_function = CustomMaskedLoss() if not is_bs_finetune else nn.BCEWithLogitsLoss()

    iter_num = 0
    epoch = 0
    if prev_iter_num and not is_finetune:
        iter_num = prev_iter_num

    t0 = time.time()

    if eval_oct_leakage or eval_mikhail_set or eval_april_leakage:
        if gpu_id == 0:
            if eval_april_leakage:
                print("evaluating on april leakage set...")
                validation_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/april_leakage_brand.tsv"
            else:
                validation_file = "/scratch/workspaceblobstore/spam/data/text/test_leakage_data_512x4.tsv" if eval_oct_leakage else "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/mikhail_bs/mikhail_set_evaluation.tsv"
            print("evaluating on validation set: ", validation_file)
            validation_set = SpamMonoDataset(positive_file=validation_file, shuffle=False)
            validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                           num_workers=4)
            eval_binary_classifier(model, validation_loader, 0)
            # else:
            #     torch.distributed.barrier()
            print("Finished evaluating on validation set. Exiting.")

        if eval_only:
            return

    if is_finetune:
        # assert that --resume_from_checkpoint is set in the system arguments.
        assert "--resume_from_checkpoint" in sys.argv, "finetuning requires --resume_from_checkpoint to load pretrained model"

        if is_bs_finetune:
            train_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/mikhail_bs/mikhail_set_validation_copy.tsv"
            print("finetuning on mikhail set: ", train_file)
            train_dataset = SpamMonoDataset(positive_file=train_file, shuffle=True, train=True)
            validation_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/mikhail_bs/mikhail_set_evaluation.tsv"
            valid_dataset = validation_set = SpamMonoDataset(positive_file=validation_file, shuffle=False)
            validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                           num_workers=4)

        else:
            train_pos_file = os.path.join(data_dir, "positives_finetune_train.tsv")
            valid_pos_file = os.path.join(data_dir, "positives_finetune_validation.tsv")
            train_neg_file = os.path.join(data_dir, "negatives_27M_train.tsv")
            valid_neg_file = os.path.join(data_dir, "negatives_27M_valid.tsv")

            # load the data
            train_dataset = SpamDataset(train_pos_file, train_neg_file, shuffle=True)
            valid_dataset = SpamDataset(valid_pos_file, valid_neg_file, shuffle=False)

            # load mikahil validation set
            validation_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/mikhail_bs/mikhail_set_evaluation.tsv"
            validation_set = SpamMonoDataset(positive_file=validation_file, shuffle=False)
            validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                           num_workers=4)

            # eval april leakage set
            validation_april_leakage_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/april_leakage_brand.tsv"
            validation_april_leakage_set = SpamMonoDataset(positive_file=validation_april_leakage_file, shuffle=False)
            validation_april_leakage_loader = DataLoader(validation_april_leakage_set, batch_size=batch_size,
                                                         shuffle=True, pin_memory=True, num_workers=4)

        print("train data pipeline counts: ", train_dataset.pipeline_counts)
        print("valid data pipeline counts: ", valid_dataset.pipeline_counts)
        print("Mikhail validation set length: ", len(validation_set))

    else:
        train_pos_file = os.path.join(data_dir, "positives_train.tsv")
        valid_pos_file = os.path.join(data_dir, "positives_valid.tsv")
        train_neg_file = os.path.join(data_dir, "negatives_train_.tsv")
        valid_neg_file = os.path.join(data_dir, "negatives_valid.tsv")

        # load the data
        train_dataset = SpamDataset(train_pos_file, train_neg_file, shuffle=True)
        valid_dataset = SpamDataset(valid_pos_file, valid_neg_file, shuffle=False)

    print("loaded datasets for training and validation")

    num_samples_seen_per_pipeline = defaultdict(int)

    if prev_num_samples_seen_per_pipeline and not is_finetune:
        num_samples_seen_per_pipeline = prev_num_samples_seen_per_pipeline

    # Calculate weights for each sample based on their subtype
    # check if weight can be loaded from saved file
    if not is_bs_finetune:
        weights_path = os.path.join(data_dir,
                                    f"weights_all_{train_pos_file.split('/')[-1].split('.')[0]}_{train_neg_file.split('/')[-1].split('.')[0]}_{sum(desired_samples.values())}.pkl")
        print("weights path: ", weights_path)
        if os.path.exists(weights_path):
            print("Loading weights from : ", weights_path)
            with open(weights_path, 'rb') as f:
                weights = pickle.load(f)
        else:
            print("Creating weights for each sample")
            weights = []
            # Calculate the total desired samples per epoch
            for num_samples, (_, _, _, pipeline) in enumerate(train_dataset):
                # get subtype from mapping. If all zeros, then it is "Negative".
                subtypes = [spam_subtype_mapping[i] for i, v in enumerate(pipeline) if v == 1]
                if len(subtypes) == 0:
                    subtypes = ["Negative"]
                try:
                    weight = sum(
                        [desired_samples[subtype] / train_dataset.pipeline_counts[subtype] for subtype in subtypes])
                except:
                    print("subtypes: ", subtypes)
                    print("pipeline_counts: ", train_dataset.pipeline_counts)
                    print("desired_samples: ", desired_samples)
                    raise Exception("Error in calculating weights")
                weights.append(weight)
                if num_samples % 100000 == 0:
                    print("num_samples processed for weights: ", num_samples)
                # du_ = torch.rand((100, 100)).to(device) * torch.rand((100, 100)).to(device)

            # save weights to file
            with open(weights_path, 'wb') as f:
                pickle.dump(weights, f)

        print("Length of weights: ", len(weights))
        print("Length of train_dataset: ", len(train_dataset))

        # Create a WeightedRandomSampler instance
        sampler_ = CustomWeightedRandomSampler(weights, len(train_dataset), replacement=True)
        sampler = DistributedProxySampler(sampler_, rank=gpu_id, num_replicas=world_size)

        # Create a DataLoader with the created sampler
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)
        train_loader.sampler.set_epoch(epoch)

        # create a second loader for brand stuffing combined tuning.
        train_bs_file = "/scratch/singularity_webdata_ws01_eastus2_nfs/rmayuranath/data/v2/finetune/mikhail_bs/mikhail_set_validation_copy.tsv"
        train_bs_dataset = SpamMonoDataset(positive_file=train_bs_file, shuffle=True, train=True)
        train_bs_loader = DataLoader(train_bs_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                     num_workers=4)

        def cycle(loader):
            while True:
                for data in loader:
                    yield data

        train_bs_loader = cycle(train_bs_loader)
        train_loader = cycle(train_loader)

    else:
        # Create a DataLoader with sampler for training
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # Create a DataLoader without sampler for validation
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    print("Loaded {} training samples".format(len(train_dataset)))
    print("Loaded {} validation samples".format(len(valid_dataset)))

    while True:

        if iter_num * batch_size == sum(desired_samples.values()):
            epoch += 1
            train_loader.sampler.set_epoch(epoch)

        candidates = [train_loader, train_bs_loader] if iter_num % 50 == 0 else [train_loader]
        # candidates = [train_loader]

        for loader in candidates:

            batch = next(loader)

            # if loader == train_bs_loader:
            #     print("training on brand stuffing data...")
            #     print("batch_size: ", batch_size)
            #     print("iter_num: ", iter_num)
            #     print("Num samples seen in batch: ", len(batch[0]))

            # determine the learning rate for this iteration
            if decay_lr:
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = learning_rate

            # evaluate on val if gpu_id == 0
            if gpu_id == 0 and iter_num % eval_interval == 0 and iter_num != 0:
                eval_binary_classifier(model, valid_loader, iter_num)

                # eval on mikhail validation set also every 2nd eval_interval
                if iter_num % (eval_interval * 2) == 0 and is_finetune and not is_bs_finetune:
                    print("evaluating on mikhail set...")
                    eval_binary_classifier(model, validation_loader, iter_num, is_mikhail=True)
                    print("evaluating on april leakage set...")
                    eval_binary_classifier(model, validation_april_leakage_loader, iter_num)

                if eval_only:
                    return

                raw_model = model.module if hasattr(model, "module") else model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'lr': 'lr',
                    'num_samples_seen_per_pipeline': num_samples_seen_per_pipeline,
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

            loss = loss_function(scores, labels.float(), pipelines) if not is_bs_finetune else loss_function(scores,
                                                                                                             labels.float()[
                                                                                                             :, 0])

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
                    num_examples_seen = iter_num * batch_size * world_size
                    writer.add_scalar('train/num_examples_seen', num_examples_seen, iter_num)
                except:
                    print("Error logging weights to tensorboard for iter_num: ", iter_num)

                # log number of samples seen per pipeline
                for idx, num_samples in num_samples_seen_per_pipeline.items():
                    writer.add_scalar(f'train/num_samples_seen_per_pipeline/{spam_subtype_mapping[idx]}', num_samples,
                                      iter_num)
                    print(f"iter {iter_num}: num_samples_seen for {spam_subtype_mapping[idx]}: {num_samples}")

            iter_num += 1


if __name__ == "__main__":
    batch_size = int(sys.argv[sys.argv.index("--batch_size") + 1]) if "--batch_size" in sys.argv else 128
    print("batch_size: ", batch_size)
    train(model, batch_size, data_dir)
    exit(0)

