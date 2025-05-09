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
from torch.distributed import init_process_group
import torch.nn.init as init
from config import try_model_names

EMBEDDING_DIM = 768
NUM_SEQ, MIN_SEQ_LENGTH = 4, 512


class PositionalEncoding(nn.Module):
    """
    Positional encodings are added to the input embeddings to provide information about the position of tokens
    in a sequence, as transformers themselves do not have a built-in notion of order
    """

    def __init__(self, seq_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.seq_length = seq_length  # 2, the maximum length of the input sequence
        self.embedding_dim = embedding_dim  # 768

        # create the positional encoding table,  pe will store the positional encodings for each token
        pe = torch.zeros(seq_length, embedding_dim)  # torch.Size([2, 768]),
        # creates a tensor (position) representing the positions of tokens in the sequence. position shape: (2,1)
        # .unsqueeze(1): add an extra dimension to the tensor, converting it from shape(seq_length,) to (seq_length,1)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  # position: tensor([[0.], [1.]])
        # div_term: Computes a scaling factor used to generate the sin and cos positional encodings
        # torch.arange(0, embedding_dim, 2): generates a tensor with even indices from 0 to embedding_dim - 1 (step=2)
        # this is because positional encodings alternate between sin and cos functions
        # (-math.log(10000.0)/embedding_dim): Computes a scaling factor based on the embedding dim. 10000.0 is a
        # constant used in the original transformer paper to control the frequency of the sin functions
        # torch.exp():Applies the exponential function to the scaled indices, resulting in a tensor of shape(emb_dim//2)
        # div_term.shape: (384,)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        # computes the sin component of the positional encoding for even indices of the embedding dimension
        # position * div_term: multiplies the position tensor (shape (seq_length, 1)) by the scaling factor
        # (shape (embedding_dim//2,)). Broadcasting ensures this operation works correctly
        # pe.shape: (2, 768), position: (2,1), div_term: (384,), position*div_term: [2,384]
        pe[:, 0::2] = torch.sin(position * div_term)
        # computes the cosine component of the positional encoding for odd indices of the embedding dimension
        pe[:, 1::2] = torch.cos(position * div_term)
        # registers the positional encoding tensor pe as a buffer in the PyTorch module
        # register_buffer: add the tensor pe to the module's state but does not treat it as a trainable parameter
        # (it will not be updated during backpropagation)
        # this ensures that the positional encoding table is saved and loaded correctly with the model.
        self.register_buffer('pe', pe)  # pe is registered as a on-trainable buffer in the PyTorch module.

    def forward(self, x):
        # add the positional encoding to the input tensor
        # x: [4,2,768], pe: [2,768], x.size(1): 2, tmp=self.pe[:2, :] is pe, so tmp.shape: [2,768]
        # Broadcasting Rules: PyTorch follows NumPy-style broadcasting rules: pytorch will automatically expand the
        # smaller tensor (tmp) to match the shape of the larger tensor (x) if their dimensions are compatible
        # here is from right to left to match: the last two dimensions of tmp([2,768]) match the last two dimensions
        # of x ([2,768]), the first dim of x (4) has no corresponding dim in tmp,
        # so pytorch will broadcast tmp along this dim
        x = x + self.pe[:x.size(1), :]  # pe: [seq_length, embedding_dim]
        return x


class TeacherModelV2(nn.Module):
    """
    Teacher model
    """

    def __init__(self, model_name="bert-base-multilingual-cased", url_layer_num=1, text_layer_num=1, combine_layer_num=1):
        super(TeacherModelV2, self).__init__()
        self.average_embeddings = True

        # keep only the first n encoder layer for url_enc/text_enc/combine_enc
        # here not initialize the embedding layer of the encoder, but instead used the params from the pretrained model.
        # Note: for student model, the embedding layer will be initialized since it needs to derive params from teacher.
        self.url_enc = BertModel.from_pretrained(model_name)
        self.url_enc.encoder.layer = self.url_enc.encoder.layer[:url_layer_num]

        self.text_enc = BertModel.from_pretrained(model_name)
        self.text_enc.encoder.layer = self.text_enc.encoder.layer[:text_layer_num]

        self.combine_enc = BertModel.from_pretrained(model_name)
        self.combine_enc.encoder.layer = self.combine_enc.encoder.layer[:combine_layer_num]
        # set to empty so dont waste space (self define Position embedding for the combine encoder)
        # only need to assign new values to word_embeddings and position_embeddings, other embedding config keep same
        self.combine_enc.embeddings.word_embeddings = None  # original value is: Embedding(119547, 768, padding_idx=0)
        self.combine_enc.embeddings.postition_embeddings = PositionalEncoding(NUM_SEQ + 1, EMBEDDING_DIM)
        self.combine_enc.apply(self._init_weights) # init all nn.Linear/nn.Embedding/nn.LayerNorm modules in combine_enc

        self.fc = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 256),  # (768,256)
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 means 5 positive categories
        )
        self.fc.apply(self._init_weights)

    def _init_weights(self, module):
        """
        module:
        BertEmbeddings(
            (word_embeddings): None, for BertModel: Embedding(119547, 768, padding_idx=0)
            (position_embeddings): PositionalEncoding(), for BertModel, it's Embedding(512, 768)
            (token_type_embeddings): Embedding(2, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        """
        # print(f'module type:{type(module)}')
        """
        module type:<class 'torch.nn.modules.sparse.Embedding'>
        module type:<class 'torch.nn.modules.sparse.Embedding'>
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class '__main__.PositionalEncoding'>
        module type:<class 'transformers.models.bert.modeling_bert.BertEmbeddings'>
        
        # encoder start
        # attention start
        module type:<class 'torch.nn.modules.linear.Linear'>    query: (768, 768)
        module type:<class 'torch.nn.modules.linear.Linear'>    key: (768, 768)
        module type:<class 'torch.nn.modules.linear.Linear'>    value: (768, 768)
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class 'transformers.models.bert.modeling_bert.BertSdpaSelfAttention'>
        
        module type:<class 'torch.nn.modules.linear.Linear'>    dense: (768, 768)
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>  (768,)
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
        module type:<class 'transformers.models.bert.modeling_bert.BertAttention'>
        # attention end
        
        module type:<class 'torch.nn.modules.linear.Linear'>            (in_features=768, out_features=3072)
        module type:<class 'transformers.activations.GELUActivation'>
        module type:<class 'transformers.models.bert.modeling_bert.BertIntermediate'>
        
        module type:<class 'torch.nn.modules.linear.Linear'>            (in_features=3072, out_features=768)
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>  (768,)
        module type:<class 'torch.nn.modules.dropout.Dropout'>          (p=0.1)
        module type:<class 'transformers.models.bert.modeling_bert.BertOutput'>
        
        module type:<class 'transformers.models.bert.modeling_bert.BertLayer'>
        module type:<class 'torch.nn.modules.container.ModuleList'>
        module type:<class 'transformers.models.bert.modeling_bert.BertEncoder'>
        # encoder end
        
        # pooler start
        module type:<class 'torch.nn.modules.linear.Linear'>     (in_features=768, out_features=768, bias=True)
        module type:<class 'torch.nn.modules.activation.Tanh'>
        module type:<class 'transformers.models.bert.modeling_bert.BertPooler'>
        # pooler end
        
        module type:<class 'transformers.models.bert.modeling_bert.BertModel'>
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, url_tokens, text_tokens):
        # get url embedding, url_emb: [2, 768], self.url_enc()['last_hidden_state'].shape: [2, 512, 768]
        url_emb = self.url_enc(input_ids=url_tokens, attention_mask=url_tokens != 0)['pooler_output']
        # get text embedding, text_tokens: [2,4,512], text_emb: [2,4,768]
        text_emb = torch.stack([self.text_enc( # for each i, the output of self.text_enc(..)['pooler_output'] is [2,768]
                input_ids=text_tokens[:, i, :],
                attention_mask=text_tokens[:, i, :] != 0)['pooler_output'] for i in range(text_tokens.shape[1])], dim=1)
        # concat url and text embedding
        # url_emb.unsqueeze(1).shape is [2, 1, 768], full_emb: [2, 5, 768]
        full_emb = torch.cat([url_emb.unsqueeze(1), text_emb], dim=1)

        # add combiner position embeddings before passing to combiner
        full_emb = self.combine_enc.embeddings.postition_embeddings(full_emb)
        # pass through LayerNorm. This will normalize across the embedding dimension
        full_emb = self.combine_enc.embeddings.LayerNorm(full_emb)
        # dropout
        full_emb = self.combine_enc.embeddings.dropout(full_emb)
        # there's no 'pooler_output' layer for the output of self.combine_enc.encoder, only has 'last_hidden_state'
        full_emb = self.combine_enc.encoder(full_emb)['last_hidden_state']

        # average across the sequence dimension, full_emb: [2, 5, 768], out_emb: [2, 768]
        out_emb = torch.mean(full_emb, dim=1) if self.average_embeddings else full_emb[:, 0, :]
        out_256d = None

        for idx, layer in enumerate(self.fc):
            out_emb = layer(out_emb)
            if idx == 0:
                out_256d = out_emb
        # out_emb is the classification result, out_256d is embedding with 256d
        # out_emb: [2,5], out_256d: [2, 256]
        return out_emb, out_256d

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        this function is unfortunately doing something very simple and is being very defensive:
        we are separating out all parameters of the model into two buckets:
        those that will experience weight decay for regularization and those that won't
        (biases, and layernorm/embedding weights). Then return the AdamW optimizer in PyTorch with custom weight decay
        settings for different groups of model parameters.
        This is a common practice in deep learning to apply weight decay selectively, as not all parameters benefit
        from regularization in the same way.
        Usage scenarios:
        1) Selective Weight Decay
            a) Some parameters, such as those in normalization layers (LayerNorm,BatchNorm) or embedding layers,
            should not be regularized because they are not part of the feature learning process
            b) Other parameters, such as weights in Linear or Conv layers, benefit from weight decay
            to prevent overfitting
        2) Fine-Tuning Pretrained Models,
            when fine-tuning pretrained models, you may want to apply weight decay only to certain layers
            (e.g. the newly added layers - combine_enc) while excluding others (e.g. pretrained embeddings).
        2) Custom Optimization Strategies
            If your model has specific layers or components that require different optimization strategies,
            this approach allows you to customize the optimizer behavior for each group of parameters.
        3)Improving Model Generalization
            By applying weight decay selectively, you can improve the model's generalization performance
            without negatively affecting parameters that should not be regularized
        """
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear,                # Fully connected layers
                                    torch.nn.Conv1d,                # 1D convolutional layers
                                    torch.nn.Conv2d,
                                    torch.nn.Conv3d,
                                    torch.nn.ConvTranspose1d,       # 1D transposed convolutional layers
                                    torch.nn.ConvTranspose2d,
                                    torch.nn.ConvTranspose3d,
                                    torch.nn.MultiheadAttention)    # Attention layers (query, key, value weights)
        blacklist_weight_modules = (torch.nn.LayerNorm,      # Layer normalization
                                    torch.nn.BatchNorm1d,    # 1D batch normalization
                                    torch.nn.BatchNorm2d,
                                    torch.nn.BatchNorm3d,
                                    torch.nn.GroupNorm,      # Group normalization
                                    torch.nn.InstanceNorm1d, # 1D instance normalization
                                    torch.nn.InstanceNorm2d,
                                    torch.nn.InstanceNorm3d,
                                    torch.nn.Embedding,      # Embedding layers
                                    torch.nn.EmbeddingBag,   # Embedding layers
                                    torch.nn.PReLU,          # Parametric ReLU (learnable parameter)
                                    torch.nn.LSTM,           # LSTM layers (biases and hidden weights)
                                    torch.nn.GRU,            # GRU layers (biases and hidden weights)
                                    torch.nn.RNN,)           # RNN layers (biases and hidden weights)

        # module_name and module are recursive layer by layer, for example:
        # module_name: url_enc, module: BertModel
        # module_name: url_enc.embeddings, module: BertEmbeddings
        # module_name: url_enc.embeddings.word_embeddings, module: Embedding
        for module_name, module in self.named_modules():
            # param_name: {weight, bias}
            for param_name, param in module.named_parameters(): # pn: param_name, p: param
                fpn = '%s.%s' % (module_name, param_name) if module_name else param_name  # full param name
                # skip frozen parameters
                if not param.requires_grad:
                    continue
                # random note: because named_modules and named_parameters are recursive, we will see the same tensors p
                # many many times, but doing it this way allows us to know which parent module any tnesor p belongs to..
                if param_name.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    continue
                if param_name.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif param_name.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # else:
                #     raise ValueError(f'the error module type is: {type(module)}')
        # validate that we considered every parameter
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


class StudentModelV2(nn.Module):
    """
    Student model
    """
    def __init__(self, model_name="prajjwal1/bert-tiny", url_layer_num=1, text_layer_num=2, combine_layer_num=1):
        super(StudentModelV2, self).__init__()
        self.average_embeddings = True

        # keep only the first n encoder layer for url_enc/text_enc/combine_enc
        # here not initialize the embedding layer of the encoder, but instead used the params from the pretrained model.
        # Note: for student model, the embedding layer will be initialized since it needs to derive params from teacher.
        self.url_enc = AutoModel.from_pretrained(model_name)
        self.url_enc.encoder.layer = self.url_enc.encoder.layer[:url_layer_num]
        self.url_enc.embeddings.word_embeddings = nn.Embedding(119547, 128)  # fill in your model embedding shape
        self.url_enc.embeddings.apply(self._init_weights)      # student retrain

        self.text_enc = AutoModel.from_pretrained(model_name)
        self.text_enc.encoder.layer = self.text_enc.encoder.layer[:text_layer_num]
        self.text_enc.embeddings.word_embeddings = nn.Embedding(119547, 128)
        self.text_enc.embeddings.apply(self._init_weights)

        self.combine_enc = BertModel.from_pretrained(model_name)
        self.combine_enc.encoder.layer = self.combine_enc.encoder.layer[:combine_layer_num]
        # set to empty so dont waste space (self define Position embedding for the combine encoder)
        # only need to assign new values to word_embeddings and position_embeddings, other embedding config keep same
        self.combine_enc.embeddings.word_embeddings = None  # original value is: Embedding(119547, 128, padding_idx=0)
        self.combine_enc.embeddings.postition_embeddings = PositionalEncoding(NUM_SEQ + 1, 128)
        self.combine_enc.apply(self._init_weights)  # init all nn.Linear/nn.Embedding/nn.LayerNorm modules in combine_enc

        self.fc = nn.Sequential(
            nn.Linear(128, 256),  # (128,256)
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 means 5 positive categories, if only 1 positive category, just set it to 1
        )
        self.fc.apply(self._init_weights)

    def _init_weights(self, module):
        """
        module:
        BertEmbeddings(
            (word_embeddings): None, for BertModel: Embedding(119547, 128, padding_idx=0)
            (position_embeddings): PositionalEncoding(), for BertModel, it's Embedding(512, 128)
            (token_type_embeddings): Embedding(2, 128)
            (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        """
        # print(f'module type:{type(module)}')
        """
        module type:<class 'torch.nn.modules.sparse.Embedding'>
        module type:<class 'torch.nn.modules.sparse.Embedding'>
        module type:<class 'torch.nn.modules.sparse.Embedding'>
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class '__main__.PositionalEncoding'>
        module type:<class 'transformers.models.bert.modeling_bert.BertEmbeddings'>

        # encoder start
        # attention start
        module type:<class 'torch.nn.modules.linear.Linear'>    query: (128, 128)
        module type:<class 'torch.nn.modules.linear.Linear'>    key: (128, 128)
        module type:<class 'torch.nn.modules.linear.Linear'>    value: (128, 128)
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class 'transformers.models.bert.modeling_bert.BertSdpaSelfAttention'>

        module type:<class 'torch.nn.modules.linear.Linear'>    dense: (128, 128)
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>  (128,)
        module type:<class 'torch.nn.modules.dropout.Dropout'>
        module type:<class 'transformers.models.bert.modeling_bert.BertSelfOutput'>
        module type:<class 'transformers.models.bert.modeling_bert.BertAttention'>
        # attention end

        module type:<class 'torch.nn.modules.linear.Linear'>            (in_features=128, out_features=512)
        module type:<class 'transformers.activations.GELUActivation'>
        module type:<class 'transformers.models.bert.modeling_bert.BertIntermediate'>

        module type:<class 'torch.nn.modules.linear.Linear'>            (in_features=512, out_features=128)
        module type:<class 'torch.nn.modules.normalization.LayerNorm'>  (128,)
        module type:<class 'torch.nn.modules.dropout.Dropout'>          (p=0.1)
        module type:<class 'transformers.models.bert.modeling_bert.BertOutput'>

        module type:<class 'transformers.models.bert.modeling_bert.BertLayer'>
        module type:<class 'torch.nn.modules.container.ModuleList'>
        module type:<class 'transformers.models.bert.modeling_bert.BertEncoder'>
        # encoder end

        # pooler start
        module type:<class 'torch.nn.modules.linear.Linear'>     (in_features=128, out_features=128, bias=True)
        module type:<class 'torch.nn.modules.activation.Tanh'>
        module type:<class 'transformers.models.bert.modeling_bert.BertPooler'>
        # pooler end

        module type:<class 'transformers.models.bert.modeling_bert.BertModel'>
        """
        print(type(module))
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, url_tokens, text_tokens):
        # get url embedding, url_emb: [2, 128], self.url_enc()['last_hidden_state'].shape: [2, 512, 128]
        url_emb = self.url_enc(input_ids=url_tokens, attention_mask=url_tokens != 0)['pooler_output']
        # get text embedding, text_tokens: [2,4,512], text_emb: [2,4,128]
        text_emb = torch.stack([self.text_enc( # for each i, the output of self.text_enc(..)['pooler_output'] is [2,128]
                input_ids=text_tokens[:, i, :],
                attention_mask=text_tokens[:, i, :] != 0)['pooler_output'] for i in range(text_tokens.shape[1])], dim=1)
        # concat url and text embedding
        # url_emb.unsqueeze(1).shape is [2, 1, 128], full_emb: [2, 5, 128]
        full_emb = torch.cat([url_emb.unsqueeze(1), text_emb], dim=1)

        # add combiner position embeddings before passing to combiner
        full_emb = self.combine_enc.embeddings.postition_embeddings(full_emb)
        # pass through LayerNorm. This will normalize across the embedding dimension
        full_emb = self.combine_enc.embeddings.LayerNorm(full_emb)
        # dropout
        full_emb = self.combine_enc.embeddings.dropout(full_emb)
        # there's no 'pooler_output' layer for the output of self.combine_enc.encoder, only has 'last_hidden_state'
        full_emb = self.combine_enc.encoder(full_emb)['last_hidden_state']

        # average across the sequence dimension, full_emb: [2, 5, 128], out_emb: [2, 128]
        out_emb = torch.mean(full_emb, dim=1) if self.average_embeddings else full_emb[:, 0, :]
        out_256d = None

        for idx, layer in enumerate(self.fc):
            out_emb = layer(out_emb)
            if idx == 0:
                out_256d = out_emb
        # out_emb is the classification result, out_256d is embedding with 256d
        # out_emb: [2,5], out_256d: [2, 256]
        return out_emb, out_256d

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        this function is unfortunately doing something very simple and is being very defensive:
        we are separating out all parameters of the model into two buckets:
        those that will experience weight decay for regularization and those that won't
        (biases, and layernorm/embedding weights). Then return the AdamW optimizer in PyTorch with custom weight decay
        settings for different groups of model parameters.
        This is a common practice in deep learning to apply weight decay selectively, as not all parameters benefit
        from regularization in the same way.
        Usage scenarios:
        1) Selective Weight Decay
            a) Some parameters, such as those in normalization layers (LayerNorm,BatchNorm) or embedding layers,
            should not be regularized because they are not part of the feature learning process
            b) Other parameters, such as weights in Linear or Conv layers, benefit from weight decay
            to prevent overfitting
        2) Fine-Tuning Pretrained Models,
            when fine-tuning pretrained models, you may want to apply weight decay only to certain layers
            (e.g. the newly added layers - combine_enc) while excluding others (e.g. pretrained embeddings).
        2) Custom Optimization Strategies
            If your model has specific layers or components that require different optimization strategies,
            this approach allows you to customize the optimizer behavior for each group of parameters.
        3)Improving Model Generalization
            By applying weight decay selectively, you can improve the model's generalization performance
            without negatively affecting parameters that should not be regularized
        """
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear,                # Fully connected layers
                                    torch.nn.Conv1d,                # 1D convolutional layers
                                    torch.nn.Conv2d,
                                    torch.nn.Conv3d,
                                    torch.nn.ConvTranspose1d,       # 1D transposed convolutional layers
                                    torch.nn.ConvTranspose2d,
                                    torch.nn.ConvTranspose3d,
                                    torch.nn.MultiheadAttention)    # Attention layers (query, key, value weights)
        blacklist_weight_modules = (torch.nn.LayerNorm,      # Layer normalization
                                    torch.nn.BatchNorm1d,    # 1D batch normalization
                                    torch.nn.BatchNorm2d,
                                    torch.nn.BatchNorm3d,
                                    torch.nn.GroupNorm,      # Group normalization
                                    torch.nn.InstanceNorm1d, # 1D instance normalization
                                    torch.nn.InstanceNorm2d,
                                    torch.nn.InstanceNorm3d,
                                    torch.nn.Embedding,      # Embedding layers
                                    torch.nn.EmbeddingBag,   # Embedding layers
                                    torch.nn.PReLU,          # Parametric ReLU (learnable parameter)
                                    torch.nn.LSTM,           # LSTM layers (biases and hidden weights)
                                    torch.nn.GRU,            # GRU layers (biases and hidden weights)
                                    torch.nn.RNN,)           # RNN layers (biases and hidden weights)

        # module_name and module are recursive layer by layer, for example:
        # module_name: url_enc, module: BertModel
        # module_name: url_enc.embeddings, module: BertEmbeddings
        # module_name: url_enc.embeddings.word_embeddings, module: Embedding
        for module_name, module in self.named_modules():
            # param_name: {weight, bias}
            for param_name, param in module.named_parameters(): # pn: param_name, p: param
                fpn = '%s.%s' % (module_name, param_name) if module_name else param_name  # full param name
                # skip frozen parameters
                if not param.requires_grad:
                    continue
                # random note: because named_modules and named_parameters are recursive, we will see the same tensors p
                # many many times, but doing it this way allows us to know which parent module any tnesor p belongs to..
                if param_name.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    continue
                if param_name.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif param_name.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                # else:
                #     raise ValueError(f'the error module type is: {type(module)}')
        # validate that we considered every parameter
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


"""
****************************************************** Unit tests ******************************************************
"""


def test_positional_enoding():
    """
    Test PositionalEncoding
    """
    batch_size = 4
    seq_length = 2
    embedding_dim = 768

    dummy_input = torch.zeros(batch_size, seq_length, embedding_dim)  # (4,2,768)
    positional_encoding = PositionalEncoding(seq_length, embedding_dim)
    print(f'Positional encoding table shape: {positional_encoding.pe.shape}')  # [2, 768]

    output = positional_encoding(dummy_input)
    print(f'Output after adding positional encodeing: {output.shape}')

    assert output.shape == (batch_size, seq_length, embedding_dim), "Output shape is incorrect!"
    print('Positional encoding test is completed!')


def test_teacher_model():
    # batch_size, seq_length
    url_tokens_shape = (2, 512,)
    text_tokens_shape = (2, 4, 512)

    url_tokens = torch.randint(0, 1000, url_tokens_shape)
    text_tokens = torch.randint(0, 1000, text_tokens_shape)
    model = TeacherModelV2(try_model_names['multilingual_bert'], url_layer_num=1, text_layer_num=1, combine_layer_num=1)
    output, output_256d = model(url_tokens, text_tokens) # out_emb: [2,5], out_256d: [2, 256]
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=2e-5, betas=(0.9, 0.95))
    assert output.shape == (2, 5) and output_256d.shape == (2, 256) and optimizer is not None, \
        "Output shape is incorrect!"
    print('Teacher model test is completed!')


def test_student_model():
    # batch_size, seq_length
    url_tokens_shape = (2, 512,)
    text_tokens_shape = (2, 4, 512)

    url_tokens = torch.randint(0, 1000, url_tokens_shape)
    text_tokens = torch.randint(0, 1000, text_tokens_shape)
    model = StudentModelV2(try_model_names['tiny_bert'], url_layer_num=1, text_layer_num=2, combine_layer_num=1)
    output, output_256d = model(url_tokens, text_tokens)  # out_emb: [2,5], out_256d: [2, 256]
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=2e-5, betas=(0.9, 0.95))
    assert output.shape == (2, 5) and output_256d.shape == (2, 256) and optimizer is not None, \
        "Output shape is incorrect!"
    print('Student model test is completed!')



if __name__ == '__main__':
    # Test PositionalEncoding
    # test_positional_enoding()

    # Test teacher model
    # test_teacher_model()

    # Test student model
    test_student_model()