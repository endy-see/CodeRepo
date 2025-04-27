#!/usr/bin/env python
# Add NCCL optimization settings
import os
import warnings

# Suppress TorchDynamo warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_IB_TIMEOUT"] = "3600"  # only increased timeout is meaningless, because if the root cause is not resolved, the program will eventually crash
os.environ["NCCL_SOCKET_NTHREADS"] = "8"
os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"  # Disable debug for production
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1048576"  # Enable FlightRecorder

import pandas as pd
import numpy as np
import torch
import sys
import signal
from datasets import Dataset
import torch.distributed as dist
from torch.distributed import ReduceOp
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from torch import nn
import torch.nn.functional as F
import pdb

class HierarchicalClassifier(PreTrainedModel):
    """Hierarchical classifier for two-level classification"""
    def __init__(self, base_model, config):
        super().__init__(config)
        self.bert = base_model
        
        # Hidden size from BERT
        hidden_size = config.hidden_size
        
        # First level classifier
        self.level1_classifier = nn.Linear(hidden_size, NUM_CLASSES['level1'])
        
        # Second level classifiers with correct output sizes
        self.sensitive_classifier = nn.Linear(hidden_size, len(SENSITIVE_VALUES))  # For sensitive values
        self.tolerant_classifier = nn.Linear(hidden_size, len(TOLERANT_VALUES))   # For tolerant values
        
        # Register label mappings as buffers
        self.register_buffer('sensitive_values', torch.tensor(SENSITIVE_VALUES, dtype=torch.long))
        self.register_buffer('tolerant_values', torch.tensor(TOLERANT_VALUES, dtype=torch.long))
        
        # Dropout
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        level2_labels=None,
        epoch=None  # Added epoch parameter for dynamic weighting
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 从last_hidden_state中选取所有batch(:)的第0个位置的token(即[CLS] token)的隐藏层状态，shape从(bs, seq_len, hidden_size)变为(bs, hidden_size)
        # 为什么用[CLS]: BERT在预训练时，[CLS] token的隐藏状态被设计为聚合整个序列的信息，常用于分类任务
        # dropout: 对[CLS]的隐藏状态应用Dropout随机置零，增强模型鲁棒性
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        # Get level1 logits
        level1_logits = self.level1_classifier(pooled_output)
        
        # Get level2 logits
        sensitive_logits = self.sensitive_classifier(pooled_output)  
        tolerant_logits = self.tolerant_classifier(pooled_output)   
        
        # 1. Calculate level1 loss with label smoothing
        level1_loss = F.cross_entropy(
            level1_logits,
            labels,
            label_smoothing=0.1
        )
        
        # 2. Initialize level2 losses
        level2_sensitive_loss = torch.tensor(0.0, device=labels.device)
        level2_tolerant_loss = torch.tensor(0.0, device=labels.device)
        
        # Handle level2 labels
        # Handle Sensitive samples (level1 label = 0)
        sensitive_mask = (labels == 0)  # 生成sensitive类别的掩码,sensitive_mask是一个布尔张量，shape与labels相同，True表示该位置是sensitive类别
        # 筛选sensitive类别的预测(sensitive_logits)和标签(level2_labels)，仅保留属于sensitive的样本的预测结果和二级标签
        if sensitive_mask.any():    # sensitive_logits.shape: (bs, 12), level2_labels.shape: (bs,)
            sensitive_logits_filtered = sensitive_logits[sensitive_mask]    # sensitive_logits_filtered是仅包含sensitive样本的logits, shape: (num_sensitive_samples, 12)
            sensitive_labels = level2_labels[sensitive_mask]                # sensitive_labels是仅包含sensitive样本的二级标签, shape: (num_sensitive_samples,)
            valid_sensitive = (sensitive_labels != -100)    # valid_sensitive是一个布尔张量，shape与sensitive_labels相同，True表示该样本的二级标签有效
            
            if valid_sensitive.any():   # 过滤无效标签(-100),进一步筛选出二级标签有效的样本
                # Map actual label values to indices 将二级标签的实际值(如100,98..])映射为类别索引(如[0,1,2])，以便计算交叉熵损失
                sensitive_indices = torch.zeros_like(sensitive_labels[valid_sensitive])     # 初始化为全0，shape与valid_sensitive相同
                for i, val in enumerate(self.sensitive_values): # 遍历sensitive_values列表
                    sensitive_indices[sensitive_labels[valid_sensitive] == val] = i # 将标签值等于val的位置赋值为当前索引i
                    
                level2_sensitive_loss = F.cross_entropy(
                    sensitive_logits_filtered[valid_sensitive],
                    sensitive_indices,
                    reduction='mean',
                    label_smoothing=0.1
                )
        
        # Handle Tolerant samples (level1 label = 2)
        tolerant_mask = (labels == 2)
        if tolerant_mask.any():
            tolerant_logits_filtered = tolerant_logits[tolerant_mask]
            tolerant_labels = level2_labels[tolerant_mask]
            valid_tolerant = (tolerant_labels != -100)
            
            if valid_tolerant.any():
                # Map actual label values to indices
                tolerant_indices = torch.zeros_like(tolerant_labels[valid_tolerant])
                for i, val in enumerate(self.tolerant_values):
                    tolerant_indices[tolerant_labels[valid_tolerant] == val] = i
                
                level2_tolerant_loss = F.cross_entropy(
                    tolerant_logits_filtered[valid_tolerant],
                    tolerant_indices,
                    reduction='mean',
                    label_smoothing=0.1
                )
        
        # 3. Dynamic loss weighting based on training progress
        if epoch is not None:
            # Start with higher weight on level1, gradually increase level2 weights
            # 在训练初期更注重一级分类，随着训练进行逐渐增加二级分类的权重
            level1_weight = max(0.5, 1.0 - epoch * 0.1)  # Gradually decrease from 1.0 to 0.5
            level2_weight = min(0.5, epoch * 0.1)  # Gradually increase from 0.0 to 0.5
            # 额外的权重调整：如果二级分类损失过大，适当降低其权重
            level2_loss_avg = (level2_sensitive_loss + level2_tolerant_loss) / 2.0
            if level2_loss_avg > level1_loss * 2:
                level2_weight *= 0.8
        else:
            level1_weight = 0.6
            level2_weight = 0.4
        
        # 4. Calculate total loss with weights
        # 使用detach()防止二次反向传播
        level2_loss = (level2_sensitive_loss + level2_tolerant_loss) / 2.0
        
        # 计算和更新EMA，但不用于损失计算
        if not torch.isnan(level2_sensitive_loss) and not torch.isnan(level2_tolerant_loss):
            current_l2_loss = level2_loss.detach()
            if hasattr(self, 'level2_loss_ema'):
                self.level2_loss_ema = 0.9 * self.level2_loss_ema + 0.1 * current_l2_loss
            else:
                self.level2_loss_ema = current_l2_loss

        # 直接计算总损失，不使用EMA进行归一化
        total_loss = level1_weight * level1_loss + level2_weight * level2_loss

        # 5. Add adaptive L2 regularization for level2 classifiers
        # 计算sensitive和tolerant分类器的L2正则化项（权重衰减）
        # 对 sensitive_classifier 和 tolerant_classifier 的所有参数（通常是权重矩阵）计算 L2 范数（即参数的平方和开根号）
        l2_reg = torch.tensor(0.0, device=labels.device)
        # 根据训练进度动态调整正则化强度
        l2_strength = 0.01 * (1.0 - level1_weight)  # 随着训练进行增加正则化强度

        # 分别计算二级分类器的正则化损失
        sensitive_l2 = sum(torch.norm(param) for param in self.sensitive_classifier.parameters())
        tolerant_l2 = sum(torch.norm(param) for param in self.tolerant_classifier.parameters())

        # 应用正则化，对大的权重施加更强的惩罚
        l2_reg = l2_strength * (sensitive_l2 + tolerant_l2)
        total_loss += l2_reg
        """正则化的原因和作用
        1. 防止过拟合
        (1)L2 正则化通过惩罚大的权重值，限制模型复杂度，避免模型过度依赖训练数据中的噪声或特定样本
        (2)数学形式：在损失函数中增加 λ∑w ^2, λ 是正则化强度系数, w 是模型参数
        2. 提高泛化能力
        正则化后的模型参数更平滑，对输入扰动更鲁棒，从而在未见过的数据上表现更好
        3. 多任务学习的稳定性
        在多任务学习中，不同任务的损失可能存在竞争关系。正则化可以平衡任务间的冲突，避免某个任务的参数主导优化过程
        """
        
        # Return losses during training
        return {
            'loss': total_loss,
            'level1_loss': level1_loss,
            'level2_sensitive_loss': level2_sensitive_loss,
            'level2_tolerant_loss': level2_tolerant_loss,
            'l2_reg_loss': l2_reg,
            'level1_weight': torch.tensor(level1_weight),
            'level2_weight': torch.tensor(level2_weight),
            'level2_loss': (level2_sensitive_loss + level2_tolerant_loss) / 2.0,
            # Raw logits with preserved dimensions
            'level1_logits': level1_logits,         
            'sensitive_logits': sensitive_logits,    
            'tolerant_logits': tolerant_logits      
        }
        
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import gc
import time
import torch._dynamo.config as dynamo_config
dynamo_config.capture_scalar_outputs = True
# Level 1 class definitions
LEVEL1_CLASSES = {
    0: 'Sensitive',  # High-risk classes
    1: 'Informational',  # No subclasses
    2: 'Tolerant'  # Low-risk classes
}

# Define level 2 label mapping
LEVEL2_MAPPING = {
    'financial credentials': 100,
    'meta credentials': 98,
    'financial registration': 96,
    'Extensive PII credentials': 94,
    'high value payment scam': 90,
    'likely software download': 87,
    'PII credentials': 85,
    'PII registration': 82,
    'non-PII credentials': 80,
    'low value payment scam': 70,
    'possible software download': 60,
    'info from known or trusted source': 30,
    'information': -100,   # Special case
    'pirate media': 5,
    'adult': 4,
    'pirate information': 3,
    'pirate software': 1,
}

# Define reverse mapping for convenience
LEVEL2_REVERSE_MAPPING = {v: k for k, v in LEVEL2_MAPPING.items()}

# Define which values belong to which level1 class
SENSITIVE_VALUES = [100, 98, 96, 94, 90, 87, 85, 82, 80, 70, 60, 30]  # Sensitive class values
TOLERANT_VALUES = [5, 4, 3, 1]  # Tolerant class values

# Define number of classes for each level
NUM_CLASSES = {
    'level1': 3,      # [0: Sensitive, 1: Informational, 2: Tolerant]
    'sensitive': 12,  # Number of sensitive subclasses
    'tolerant': 4,    # Number of tolerant subclasses
}

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)

class QueryDataset(TorchDataset):
    """Custom dataset for Query and risk_type data with optimized tokenization"""
    def __init__(self, data, tokenizer, required_columns, max_length=512, cache_dir=None, name=None):
        """
        Initialize dataset with both level 1 and level 2 labels
        required_columns: [prompt_col, collapsed_label_col, risk_type_label_col]
        """
        self.name = name
        self.tokenizer = tokenizer
        self.required_columns = required_columns
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Validate required columns
        if len(required_columns) != 3:
            raise ValueError("Exactly 3 columns required: prompt, collapsed_label, risk_type_label")
        
        # Convert data to Hugging Face dataset
        if isinstance(data, pd.DataFrame):
            # Ensure numeric type for labels
            data[required_columns[1]] = pd.to_numeric(data[required_columns[1]], errors='raise')
            data[required_columns[2]] = pd.to_numeric(data[required_columns[2]], errors='raise')
            self.dataset = Dataset.from_pandas(data)
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        else:
            self.dataset = data
            
        self.load_from_cache()
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get item with both level1 and level2 labels"""
        # Always include input_ids, attention_mask, and level1 label
        item = {
            'input_ids': self.encoded_data['input_ids'][idx],
            'attention_mask': self.encoded_data['attention_mask'][idx],
            'labels': self.level1_labels[idx],
            'level2_labels': self.level2_labels[idx]
        }
            
        return item
            
    def load_from_cache(self):
        """Load tokenized data and labels from cache directory"""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.cache_dir:
            data_path = self.cache_dir / f'{self.name}_encoded_data_rank{local_rank}.pt'
            labels_path = self.cache_dir / f'{self.name}_labels_rank{local_rank}.pt'
            
            if data_path.exists() and labels_path.exists():
                # Load tokenized data
                from transformers.tokenization_utils_base import BatchEncoding
                from tokenizers import Encoding
                with torch.serialization.safe_globals([BatchEncoding, Encoding]):
                    self.encoded_data = torch.load(data_path, weights_only=False)
                    labels_dict = torch.load(labels_path, weights_only=False)
                    
                    # Load both level1 and level2 labels
                    if isinstance(labels_dict, dict):
                        self.level1_labels = labels_dict['level1']
                        self.level2_labels = labels_dict['level2']
                    else:
                        # Backward compatibility for old cache format
                        logger.warning("Old cache format detected, clearing and rebuilding cache...")
                        raise FileNotFoundError("Old cache format")
                        
                logger.info(f"Data and labels loaded from cache for rank {local_rank}")
                return
        
        logger.warning(f"Cache not found for name {self.name} and rank {local_rank}. Now tokenizing data...")
        # Pre-tokenize data in batches
        batch_size = 4000 # Process 1000 examples at a time
        all_input_ids = []
        all_attention_masks = []
        
        # Use tqdm to show progress
        for i in tqdm(range(0, len(self.dataset), batch_size), desc="Tokenizing data"):
            batch_texts = self.dataset[self.required_columns[0]][i:i+batch_size]
            # Tokenize the batch
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # Append to lists
            all_input_ids.append(encoded['input_ids'].cpu())
            all_attention_masks.append(encoded['attention_mask'].cpu())
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % 800 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Concatenate all batches
        self.encoded_data = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
        
        # Process level 1 labels (collapsed_label: [0,1,2])
        self.level1_labels = torch.tensor(self.dataset[self.required_columns[1]], dtype=torch.long)
        
        # Process level 2 labels (risk_type_label)
        # For collapsed_label=0 (Sensitive): risk_type_label range 0-11
        # For collapsed_label=2 (Tolerant): risk_type_label range 12-15
        # For collapsed_label=1 (Informational): no level2 label needed
        self.level2_labels = torch.tensor(self.dataset[self.required_columns[2]], dtype=torch.long)
        print("--->>> self.encoded_data, self.level1_labels and self.level2_labels are ready!")
        
        # Cache both level1 and level2 labels
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.save(self.encoded_data, self.cache_dir / f'{self.name}_encoded_data_rank{local_rank}.pt')
            torch.save(
                {'level1': self.level1_labels, 'level2': self.level2_labels},
                self.cache_dir / f'{self.name}_labels_rank{local_rank}.pt'
            )
            logger.info(f"Data and labels cached for rank {local_rank}")
        
        logger.info("Labels processed and saved to cache.")
        
        # Clear temporary lists
        del all_input_ids, all_attention_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()    

def load_and_split_data(file_path, required_columns):
    """Load data and split into train/val/test sets with 8:1:1 ratio"""
    # Load the data and ensure numeric labels
    df = pd.read_csv(file_path, sep='\t')
    df[required_columns[1]] = pd.to_numeric(df[required_columns[1]], errors='raise')
    df[required_columns[2]] = pd.to_numeric(df[required_columns[2]], errors='raise')
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[required_columns]
    # Clean the data
    df[required_columns[0]] = df[required_columns[0]].fillna('')
        
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # Split the data
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    return train_df, val_df, test_df

def save_model(model, tokenizer, save_dir, accelerator):
    """Helper function to save model and tokenizer with optimized performance"""
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # First notify all processes that we're starting to save
        if accelerator.is_main_process:
            tqdm.write(f"\nStarting model save to: {save_dir}")
        accelerator.wait_for_everyone()
        
        # Get unwrapped model state dictionary
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Quick sync to ensure model state is consistent
        torch.cuda.synchronize()
        
        # Only main process saves
        if accelerator.is_main_process:
            try:
                # Save hierarchical classification metadata with label ranges
                metadata = {
                    "model_type": "HierarchicalClassifier",
                    "level1": {
                        "num_classes": NUM_CLASSES['level1'],
                        "class_mapping": LEVEL1_CLASSES
                    },
                    "level2": {
                        "label_mapping": LEVEL2_MAPPING,
                        "sensitive": {
                            "num_classes": NUM_CLASSES['sensitive'],
                            "valid_values": SENSITIVE_VALUES
                        },
                        "tolerant": {
                            "num_classes": NUM_CLASSES['tolerant'],
                            "valid_values": TOLERANT_VALUES
                        }
                    }
                }
                unwrapped_model.config.update(metadata)
                unwrapped_model.config.save_pretrained(save_dir)
                
                tokenizer.save_pretrained(
                    save_dir,
                    safe_serialization=True,
                    legacy_format=False
                )
                
                # Then save the model weights
                with torch.no_grad():
                    unwrapped_model.save_pretrained(
                        save_dir,
                        save_format="safetensors",
                        safe_serialization=True,
                        max_shard_size="100MB",  # Smaller shard size for better handling
                        safe_serialization_file_atomic=True
                    )
                
                tqdm.write(f"Model successfully saved to: {save_dir}")
                
                # Create success flag file
                (save_dir / ".save_completed").touch()
                
            except Exception as save_error:
                tqdm.write(f"Error during save: {save_error}")
                # Create error flag file
                (save_dir / ".save_failed").touch()
                raise save_error
        
        # Quick sync point
        accelerator.wait_for_everyone()
        
        # All processes check save status
        if accelerator.is_main_process:
            if (save_dir / ".save_failed").exists():
                raise RuntimeError("Model save failed")
            
        # Final sync
        accelerator.wait_for_everyone()
        return save_dir
        
    except Exception as e:
        tqdm.write(f"\nError saving model: {e}")
        # Let other processes know there was an error
        if accelerator.is_main_process:
            (save_dir / ".save_failed").touch()
        return None
    finally:
        # Clean up status files
        if accelerator.is_main_process:
            try:
                (save_dir / ".save_completed").unlink(missing_ok=True)
                (save_dir / ".save_failed").unlink(missing_ok=True)
            except:
                pass
        
def training_function(args):
    # Initialize accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device_placement=True,  # Automatically place tensors on the correct device
        split_batches=True,  # Split batches across devices
    )
    
    # Set up signal handlers for cleanup
    def cleanup():
        """Cleanup function to properly handle distributed training shutdown"""
        if accelerator.is_main_process:
            tqdm.write("\nCleaning up distributed training...")        
        try:
            # Ensure all processes are synced
            accelerator.wait_for_everyone()
            # Cleanup distributed training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Final sync
                torch.distributed.destroy_process_group()
            if accelerator.is_main_process:
                tqdm.write("Cleanup completed successfully")
        except Exception as e:
            if accelerator.is_main_process:
                tqdm.write(f"Error during cleanup: {e}")
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)    # SIGINT 是一个信号，通常通过按下 Ctrl+C 组合键发送，用于中断程序执行
    signal.signal(signal.SIGTERM, signal_handler)   # SIGTERM 是一个信号，用于请求程序终止，通常由系统或进程管理器发送
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info(accelerator.state)
    
    # Initialize tensorboard writer
    if accelerator.is_main_process:
        tensorboard_dir = Path(args.output_dir) / "tensorboard_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"Tensorboard logs will be saved to: {tensorboard_dir}")
    
    # Load and preprocess data
    logger.info("Loading dataset...")    
    # Initialize tokenizer and model
    logger.info("Loading tokenizer and model...")
    
    # Load base BERT model and tokenizer
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(model_id)
    
    # Move base model to GPU if available
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    
    # Configure and initialize hierarchical classifier
    config = base_model.config
    config.gradient_checkpointing = True
    model = HierarchicalClassifier(base_model, config)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    if accelerator.is_main_process:
        logger.info("\nInitialized Hierarchical Classifier with structure:")
        logger.info(f"  Level 1 classes: {NUM_CLASSES['level1']}")
        logger.info(f"  Level 2 - Sensitive classes: {NUM_CLASSES['sensitive']}")
        logger.info(f"  Level 2 - Tolerant classes: {NUM_CLASSES['tolerant']}")
    
    # Load and split dataset
    required_columns = args.train_columns.split(',')
    train_df, val_df, test_df = load_and_split_data(args.train_file, required_columns)
    
    # Create datasets
    train_dataset = QueryDataset(train_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='train')
    val_dataset = QueryDataset(val_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='val')
    test_dataset = QueryDataset(test_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='test')
    
    # Create DataLoaders with optimized settings
    def create_dataloader(dataset, shuffle):
        # Always drop last batch in distributed setting to ensure even batch sizes
        drop_last = accelerator.use_distributed

        # Set sampler for distributed training
        if accelerator.use_distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=args.seed
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=min(args.num_workers, 4),  # Limit workers for stability
            pin_memory=True,
            drop_last=drop_last,
            pin_memory_device='cuda' if torch.cuda.is_available() else 'cpu',
            persistent_workers=True if args.num_workers > 0 else False,
        )
    
    train_dataloader = create_dataloader(train_dataset, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, shuffle=False)
    
    # Initialize optimizer with carefully tuned paramters
    no_decay = ['bias', 'LayerNorm.weight']  # Don't apply weight decay to these parameters
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Add learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=num_training_steps,
        pct_start=warmup_steps/num_training_steps,  # 10% of training for warmup
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
        three_phase=False,
    )
    
    # Empty CUDA cache before preparing model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )
    
    # Training loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    # Evaluation function
    def evaluate(model, dataloader, split="val"):
        """Evaluate model with QPS tracking and hierarchical metrics"""
        model.eval()
        # Tracking metrics
        all_level1_preds = []
        all_level1_labels = []
        all_sensitive_preds = []
        all_sensitive_labels = []
        all_tolerant_preds = []
        all_tolerant_labels = []
        total_loss = 0
        total_samples = 0
        eval_start_time = time.time()
        
        eval_progress = tqdm(
            dataloader,
            desc=f"Evaluating {split}",
            leave=False,
            disable=not accelerator.is_local_main_process
        )
        
        for batch in eval_progress:
            with torch.no_grad():
                # Forward pass
                outputs = model(**batch)
                total_loss += outputs['loss'].detach().float()
                
                # Get predictions from model outputs
                level1_preds = outputs['level1_logits'].argmax(dim=-1)
                level1_labels = batch['labels']
                
                # Track batch size for QPS calculation
                batch_size = level1_labels.size(0)
                total_samples += batch_size * accelerator.num_processes
                
                # Gather predictions from all processes
                level1_preds = accelerator.gather(level1_preds)     
                level1_labels = accelerator.gather(level1_labels)   
                
                # Store level 1 predictions
                all_level1_preds.extend(level1_preds.cpu().numpy())
                all_level1_labels.extend(level1_labels.cpu().numpy())
                
                # Handle level 2 predictions
                # Handle level 2 predictions
                if 'level2_labels' in batch:
                    # Get raw logits
                    sensitive_logits = outputs['sensitive_logits']  
                    tolerant_logits = outputs['tolerant_logits']   
                    
                    # Gather logits first to preserve dimensions
                    gathered_sensitive_logits = accelerator.gather_for_metrics(sensitive_logits)  
                    gathered_tolerant_logits = accelerator.gather_for_metrics(tolerant_logits)    
                    
                    # Gather labels for masking
                    gathered_level2_labels = accelerator.gather_for_metrics(batch['level2_labels'])
                    gathered_level1_labels = accelerator.gather_for_metrics(batch['labels'])
                    
                    # Now get predictions from gathered logits and map back to actual label values
                    gathered_sensitive_preds = gathered_sensitive_logits.argmax(dim=-1)
                    gathered_tolerant_preds = gathered_tolerant_logits.argmax(dim=-1)

                    # Map predictions back to actual values
                    mapped_sensitive_preds = torch.zeros_like(gathered_sensitive_preds)
                    for i, val in enumerate(SENSITIVE_VALUES):
                        mapped_sensitive_preds[gathered_sensitive_preds == i] = val

                    mapped_tolerant_preds = torch.zeros_like(gathered_tolerant_preds)
                    for i, val in enumerate(TOLERANT_VALUES):
                        mapped_tolerant_preds[gathered_tolerant_preds == i] = val
                    
                    # Process gathered tensors
                    if accelerator.is_main_process:
                        # Create masks for valid predictions
                        sensitive_valid = (gathered_level1_labels == 0) & (gathered_level2_labels != -100)
                        tolerant_valid = (gathered_level1_labels == 2) & (gathered_level2_labels != -100)

                        # Process sensitive predictions
                        if sensitive_valid.any():
                            all_sensitive_preds.extend(mapped_sensitive_preds[sensitive_valid].cpu().numpy())
                            all_sensitive_labels.extend(gathered_level2_labels[sensitive_valid].cpu().numpy())
                            if accelerator.is_main_process and epoch == 0:  # 只在第一个epoch打印调试信息
                                print("\nDebug - Sensitive predictions:")
                                print(f"Number of valid sensitive samples: {sensitive_valid.sum().item()}")
                                print(f"Unique predicted values: {np.unique(mapped_sensitive_preds[sensitive_valid].cpu().numpy())}")
                                print(f"Unique label values: {np.unique(gathered_level2_labels[sensitive_valid].cpu().numpy())}")
                        
                        # Process tolerant predictions
                        if tolerant_valid.any():
                            all_tolerant_preds.extend(mapped_tolerant_preds[tolerant_valid].cpu().numpy())
                            all_tolerant_labels.extend(gathered_level2_labels[tolerant_valid].cpu().numpy())
                            if accelerator.is_main_process and epoch == 0:  # 只在第一个epoch打印调试信息
                                print("\nDebug - Tolerant predictions:")
                                print(f"Number of valid tolerant samples: {tolerant_valid.sum().item()}")
                                print(f"Unique predicted values: {np.unique(mapped_tolerant_preds[tolerant_valid].cpu().numpy())}")
                                print(f"Unique label values: {np.unique(gathered_level2_labels[tolerant_valid].cpu().numpy())}")
                    
                    # Ensure all processes are synced before continuing
                    accelerator.wait_for_everyone()

        # Calculate average loss
        avg_loss = total_loss.item() / len(dataloader)
        
        # Calculate level 1 metrics
        level1_metrics = precision_recall_fscore_support(
            all_level1_labels,
            all_level1_preds,
            average='weighted',
            zero_division=0,
            labels=list(LEVEL1_CLASSES.keys())
        )
        
        # Calculate per-class metrics for level 1
        level1_class_metrics = precision_recall_fscore_support(
            all_level1_labels,
            all_level1_preds,
            average=None,
            zero_division=0,
            labels=list(LEVEL1_CLASSES.keys())
        )

        # Calculate level 2 metrics for Sensitive class if available
        sensitive_metrics = None
        if all_sensitive_preds:
            sensitive_metrics = None
            if all_sensitive_preds and all_sensitive_labels:  # Only calculate if we have predictions
                sensitive_metrics = precision_recall_fscore_support(
                    all_sensitive_labels,
                    all_sensitive_preds,
                    average='weighted',
                    zero_division=0,
                    labels=SENSITIVE_VALUES  # Use predefined sensitive class values
                )

        # Calculate level 2 metrics for Tolerant class if available
        tolerant_metrics = None
        if all_tolerant_preds:
            tolerant_metrics = None
            if all_tolerant_preds and all_tolerant_labels:  # Only calculate if we have predictions
                tolerant_metrics = precision_recall_fscore_support(
                    all_tolerant_labels,
                    all_tolerant_preds,
                    average='weighted',
                    zero_division=0,
                    labels=TOLERANT_VALUES  # Use predefined tolerant class values
                )
        # Calculate overall QPS once
        eval_time = time.time() - eval_start_time
        qps = total_samples / eval_time
        if accelerator.is_main_process:
            logger.info(
                f"Evaluation completed: processed {total_samples} samples in {eval_time:.2f}s "
                f"({qps:.1f} samples/second)"
            )
            
        if accelerator.is_main_process:
            # Performance metrics
            eval_results = [
                f"\n{split} Evaluation Results (QPS: {qps:.1f}):",
                f"Average Loss: {avg_loss:.4f}",
                "\n=== Level 1 Classification (Primary) ===",
                f"Overall Metrics:",
                f"  F1 Score: {level1_metrics[2]:.4f}",
                f"  Precision: {level1_metrics[0]:.4f}",
                f"  Recall: {level1_metrics[1]:.4f}"
            ]
            
            # Add detailed level 1 metrics for each class
            for i, class_name in LEVEL1_CLASSES.items():
                eval_results.extend([
                    f"\n{class_name} Class:",
                    f"  F1 Score: {level1_class_metrics[2][i]:.4f}",
                    f"  Precision: {level1_class_metrics[0][i]:.4f}",
                    f"  Recall: {level1_class_metrics[1][i]:.4f}",
                    f"  Support: {level1_class_metrics[3][i]}"
                ])
            
            # Show level 2 metrics if available
            if sensitive_metrics is not None or tolerant_metrics is not None:
                eval_results.append("\n=== Level 2 Classification (Secondary) ===")
            
                if sensitive_metrics is not None:
                    eval_results.extend([
                        f"\nSensitive Class Metrics:",
                        f"  F1 Score: {sensitive_metrics[2]:.4f}",
                        f"  Precision: {sensitive_metrics[0]:.4f}",
                        f"  Recall: {sensitive_metrics[1]:.4f}",
                        f"  Support: {sum(sensitive_metrics[3]) if sensitive_metrics is not None and sensitive_metrics[3] is not None else 0}"
                    ])
                
                if tolerant_metrics is not None:
                    eval_results.extend([
                        f"\nTolerant Class Metrics:",
                        f"  F1 Score: {tolerant_metrics[2]:.4f}",
                        f"  Precision: {tolerant_metrics[0]:.4f}",
                        f"  Recall: {tolerant_metrics[1]:.4f}",
                        f"  Support: {sum(tolerant_metrics[3]) if tolerant_metrics is not None and tolerant_metrics[3] is not None else 0}"
                    ])
            
            # Write evaluation results
            tqdm.write('\n'.join(eval_results))
            
            # Log metrics to tensorboard
            writer.add_scalar(f'{split}/qps', qps, epoch)
            writer.add_scalar(f'{split}/loss', avg_loss, epoch)
            
            # Log level 1 metrics
            writer.add_scalar(f'{split}/level1/precision', level1_metrics[0], epoch)
            writer.add_scalar(f'{split}/level1/recall', level1_metrics[1], epoch)
            writer.add_scalar(f'{split}/level1/f1', level1_metrics[2], epoch)
            
            # Log per-class level 1 metrics
            for i, class_name in LEVEL1_CLASSES.items():
                writer.add_scalar(f'{split}/level1_{class_name}/precision', level1_class_metrics[0][i], epoch)
                writer.add_scalar(f'{split}/level1_{class_name}/recall', level1_class_metrics[1][i], epoch)
                writer.add_scalar(f'{split}/level1_{class_name}/f1', level1_class_metrics[2][i], epoch)
            
            # Log level 2 metrics if available
            if sensitive_metrics is not None:
                writer.add_scalar(f'{split}/level2_sensitive/precision', sensitive_metrics[0], epoch)
                writer.add_scalar(f'{split}/level2_sensitive/recall', sensitive_metrics[1], epoch)
                writer.add_scalar(f'{split}/level2_sensitive/f1', sensitive_metrics[2], epoch)
            
            if tolerant_metrics is not None:
                writer.add_scalar(f'{split}/level2_tolerant/precision', tolerant_metrics[0], epoch)
                writer.add_scalar(f'{split}/level2_tolerant/recall', tolerant_metrics[1], epoch)
                writer.add_scalar(f'{split}/level2_tolerant/f1', tolerant_metrics[2], epoch)
            # Overall F1 is the same as level1 F1 for consistency
            writer.add_scalar(f'{split}/f1', level1_metrics[2], epoch)
        
        # Return comprehensive metrics dictionary
        return {
            'loss': avg_loss,
            'qps':qps,
            'level1': {
                'precision': level1_metrics[0],
                'recall': level1_metrics[1],
                'f1': level1_metrics[2],
                'class_metrics': {
                    name: {
                        'precision': level1_class_metrics[0][i],
                        'recall': level1_class_metrics[1][i],
                        'f1': level1_class_metrics[2][i],
                        'support': level1_class_metrics[3][i]
                    } for i, name in LEVEL1_CLASSES.items()
                }
            },
            'level2_sensitive': {
                'precision': sensitive_metrics[0] if sensitive_metrics else 0.0,
                'recall': sensitive_metrics[1] if sensitive_metrics else 0.0,
                'f1': sensitive_metrics[2] if sensitive_metrics else 0.0
            } if sensitive_metrics else None,
            'level2_tolerant': {
                'precision': tolerant_metrics[0] if tolerant_metrics else 0.0,
                'recall': tolerant_metrics[1] if tolerant_metrics else 0.0,
                'f1': tolerant_metrics[2] if tolerant_metrics else 0.0
            } if tolerant_metrics else None
        }
    
    # Initialize training variables
    total_training_steps = len(train_dataloader)
    best_val_f1 = 0.0  # Track best level 1 F1 score
    best_model_path = None
    
    # Progress bar for all epochs
    progress_bar = tqdm(
        range(total_training_steps * args.num_epochs),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )
    
    for epoch in range(args.num_epochs):
        # Verify distributed training state at the start of each epoch
        accelerator.wait_for_everyone()
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed training state lost between epochs")
        
        if accelerator.is_main_process:
            tqdm.write(f"\nStarting epoch {epoch + 1}/{args.num_epochs}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        # Training loop with batch size tracking for QPS
        total_samples = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            batch_size = batch['labels'].size(0)
            total_samples += batch_size * accelerator.num_processes
            
            with accelerator.accumulate(model):
                # Forward pass with epoch information for dynamic loss weighting
                batch['epoch'] = epoch
                outputs = model(**batch)
                
                # Get individual loss components
                loss = outputs['loss'] / args.gradient_accumulation_steps
                total_loss += loss.detach().float()
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Gradient clipping with monitoring
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if grad_norm > 1.0 and accelerator.is_main_process and step % 100 == 0:
                        tqdm.write(f"\nWarning: Large gradient norm: {grad_norm:.2f}")
                    if grad_norm > 10.0:
                        optimizer.zero_grad()
                        tqdm.write(f"\nSkipping update due to large gradient: {grad_norm:.2f}")
                        continue
                    
                    # Log detailed loss components every N steps
                    if step % 100 == 0 and accelerator.is_main_process:
                        tqdm.write(
                            f"\nStep {step} Losses:"
                            f"\n  Level1: {outputs['level1_loss']:.4f}"
                            f"\n  Level2-Sensitive: {outputs['level2_sensitive_loss']:.4f}"
                            f"\n  Level2-Tolerant: {outputs['level2_tolerant_loss']:.4f}"
                            f"\n  L2-Reg: {outputs['l2_reg_loss']:.4f}"
                            f"\n  Weights: L1={outputs['level1_weight']:.2f}, L2={outputs['level2_weight']:.2f}"
                        )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log to tensorboard
                if accelerator.is_main_process and step % 10 == 0:
                    global_step = epoch * len(train_dataloader) + step
                    writer.add_scalar('training/level1_loss', outputs['level1_loss'], global_step)
                    writer.add_scalar('training/level2_sensitive_loss', outputs['level2_sensitive_loss'], global_step)
                    writer.add_scalar('training/level2_tolerant_loss', outputs['level2_tolerant_loss'], global_step)
                    writer.add_scalar('training/l2_reg_loss', outputs['l2_reg_loss'], global_step)
                    writer.add_scalar('training/level1_weight', outputs['level1_weight'], global_step)
                    writer.add_scalar('training/level2_weight', outputs['level2_weight'], global_step)
                
                # Update progress and QPS
                if step % 10 == 0:
                    current_qps = total_samples / (time.time() - start_time)
                    avg_loss = total_loss.item() / (step + 1)
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            {'Loss': f"{avg_loss:.4f}",
                            'QPS': f"{current_qps:.1f}"}
                        )
                    progress_bar.update(args.gradient_accumulation_steps)
        
        # Evaluation phase
        if accelerator.is_main_process:
            tqdm.write(f"\nEvaluating epoch {epoch + 1}/{args.num_epochs}...")
        
        accelerator.wait_for_everyone()
        
        # Get validation and test metrics
        val_metrics = evaluate(model, val_dataloader, "val")
        test_metrics = evaluate(model, test_dataloader, "test")
        
        # Get level 1 F1 score (primary metric for model selection)
        val_level1_f1 = val_metrics['level1']['f1']
        
        # Save if we have a new best level 1 F1 score
        if val_level1_f1 > best_val_f1:
            best_val_f1 = val_level1_f1
            
            # Get level 2 metrics (secondary metrics)
            val_sensitive_f1 = val_metrics['level2_sensitive']['f1'] if val_metrics['level2_sensitive'] else 0.0
            val_tolerant_f1 = val_metrics['level2_tolerant']['f1'] if val_metrics['level2_tolerant'] else 0.0
            
            # Create checkpoint directory focusing on level 1 F1
            checkpoint_dir = Path(args.output_dir) / f"best_model_l1f1_{val_level1_f1:.4f}_epoch_{epoch+1}"
            
            # Save model
            save_model(model, tokenizer, checkpoint_dir, accelerator)
            
            if accelerator.is_main_process:
                # Log metrics to tensorboard prioritizing level 1
                writer.add_scalar('best/level1_f1', val_level1_f1, epoch)
                writer.add_scalar('best/test_level1_f1', test_metrics['level1']['f1'], epoch)
                
                # Add level 2 metrics as secondary indicators
                if val_metrics['level2_sensitive']:
                    writer.add_scalar('best/level2_sensitive_f1', val_sensitive_f1, epoch)
                if val_metrics['level2_tolerant']:
                    writer.add_scalar('best/level2_tolerant_f1', val_tolerant_f1, epoch)
                
                # Log corresponding test metrics
                test_f1_level1 = test_metrics['level1']['f1']
                test_f1_sensitive = test_metrics['level2_sensitive']['f1'] if test_metrics['level2_sensitive'] else 0.0
                test_f1_tolerant = test_metrics['level2_tolerant']['f1'] if test_metrics['level2_tolerant'] else 0.0
                
                writer.add_scalar('best/test_level1_f1', test_f1_level1, epoch)
                writer.add_scalar('best/test_sensitive_f1', test_f1_sensitive, epoch)
                writer.add_scalar('best/test_tolerant_f1', test_f1_tolerant, epoch)
                
                # Print evaluation results with clear hierarchy
                tqdm.write("\nNew best model saved!")
                tqdm.write("Level 1 Classification Results (Primary Metrics):")
                tqdm.write(f"  F1 Score: {val_level1_f1:.4f}")
                tqdm.write(f"  Test F1 Score: {test_metrics['level1']['f1']:.4f}")
                
                tqdm.write("\nLevel 2 Classification Results (Secondary Metrics):")
                if val_metrics['level2_sensitive']:
                    tqdm.write("  Sensitive Class:")
                    tqdm.write(f"    F1 Score: {val_sensitive_f1:.4f}")
                    tqdm.write(f"    Test F1 Score: {test_metrics['level2_sensitive']['f1']:.4f}")
                if val_metrics['level2_tolerant']:
                    tqdm.write("  Tolerant Class:")
                    tqdm.write(f"    F1 Score: {val_tolerant_f1:.4f}")
                    tqdm.write(f"    Test F1 Score: {test_metrics['level2_tolerant']['f1']:.4f}")
                
                # Calculate and display QPS for validation and test
                # We already have the QPS from validation and test evaluations
                val_qps = val_metrics.get('qps', 0.0)
                test_qps = test_metrics.get('qps', 0.0)
                avg_eval_qps = (val_qps + test_qps) / 2 if val_qps and test_qps else 0.0
                
                tqdm.write(f"\nPerformance Metrics:")
                tqdm.write(f"  Validation QPS: {val_qps:.1f}")
                tqdm.write(f"  Test QPS: {test_qps:.1f}")
                tqdm.write(f"  Average Evaluation QPS: {avg_eval_qps:.1f}")
            
            # Save latest model
            latest_dir = Path(args.output_dir) / "latest_model"
            save_model(model, tokenizer, latest_dir, accelerator)
        
        
        # Make sure all processes are synced before next epoch
        accelerator.wait_for_everyone()           
    # except Exception as e:
    #     logger.error(f"Training failed with error: {e}")
    #     cleanup()
    #     raise
    # finally:
    #     if accelerator.is_main_process and 'writer' in locals():
    #         writer.close()
    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description="Train ModernBERT risk type classifier")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to data file with 'query' and 'risk_type' columns (tsv)")
    parser.add_argument("--train_columns", type=str, required=True,
                        help="Comma-separated list for training columns")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache tokenized data")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,  # Reduced batch size
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,  # Increased gradient accumulation
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.005,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    training_function(args)
