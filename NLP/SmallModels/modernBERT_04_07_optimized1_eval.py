#!/usr/bin/env python
import os
import warnings
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset as TorchDataset
import gc
import logging
from torch.utils.tensorboard import SummaryWriter
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define class names
id2class = {
	0:'information', 
    1:'possible software download', 
    2:'likely software download', 
    3:'PII credentials', 
    4:'pirate media', 
    5:'high value payment scam', 
    6:'PII registration', 
    7:'non-PII credentials', 
    8:'low value payment scam', 
    9:'adult', 
    10:'financial credentials', 
    11:'pirate software', 
    12:'info from known or trusted source', 
    13:'financial registration', 
    14:'Extensive PII credentials', 
    15:'meta credentials', 
    16:'pirate information'
}
label_keys = list(id2class.keys())
# Initialize accelerator for multi-GPU training
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    device_placement=True,  # Automatically place tensors on the correct device
    split_batches=True,  # Split batches across devices
)

class QueryDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        if isinstance(data, pd.DataFrame):
            self.dataset = Dataset.from_pandas(data)
        else:
            self.dataset = data
        self.max_length = max_length
        
        # Tokenize data
        print("Tokenizing data...")
        texts = self.dataset['prompt']
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        self.labels = torch.tensor(self.dataset['risk_type_label'], dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


# Import custom dataset class from training file

logger = logging.getLogger(__name__)
# Evaluation function
def evaluate(model, dataloader, split="val"):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    # Define high-level category mappings
    sensitive_classes = ['financial credentials', 'meta credentials', 'financial registration',
                        'Extensive PII credentials', 'high value payment scam', 'likely software download',
                        'PII credentials', 'PII registration', 'non-PII credentials', 'low value payment scam',
                        'possible software download', 'info from known or trusted source']
    informational_classes = ['information']
    tolerant_classes = ['pirate media', 'adult', 'pirate information', 'pirate software']
    
    # Map classes to category IDs
    sensitive_ids = [i for i, class_name in id2class.items() if class_name in sensitive_classes]
    informational_ids = [i for i, class_name in id2class.items() if class_name in informational_classes]
    tolerant_ids = [i for i, class_name in id2class.items() if class_name in tolerant_classes]
    
    # Create progress bar for evaluation
    eval_progress = tqdm(
        dataloader,
        desc=f"Evaluating {split}",
        leave=False,
        disable=not accelerator.is_local_main_process
    )
    
    total_samples = 0
    eval_start_time = time.time()
    
    # Evaluation loop
    for batch in eval_progress:
        with torch.no_grad():
            batch_size = len(batch['input_ids']) * accelerator.num_processes
            total_samples += batch_size
            
            outputs = model(**batch)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            
            # Gather predictions and labels from all processes
            predictions = accelerator.gather_for_metrics(predictions)
            labels = accelerator.gather_for_metrics(batch["labels"])
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.detach().float()
    
    eval_time = time.time() - eval_start_time
    print(f'--->>> eval_time: {eval_time:.2f} seconds, total_samples: {len(all_labels)}')
    qps = total_samples / eval_time
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算基础指标
    avg_loss = total_loss.item() / len(dataloader)
    
    # 计算三个大类别的指标
    def get_category_metrics(class_ids):
        y_true_binary = np.isin(all_labels, class_ids).astype(int)
        y_pred_binary = np.isin(all_predictions, class_ids).astype(int)
        
        # 使用binary average因为是二分类问题
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary,
            y_pred_binary,
            average='binary',
            zero_division=0
        )
        return precision, recall, f1, sum(y_true_binary)
    
    # 计算sensitive类别的指标
    sensitive_ids = [i for i, class_name in id2class.items() if class_name in sensitive_classes]
    sensitive_p, sensitive_r, sensitive_f1, sensitive_support = get_category_metrics(sensitive_ids)
    
    # 计算informational类别的指标
    informational_ids = [i for i, class_name in id2class.items() if class_name in informational_classes]
    info_p, info_r, info_f1, info_support = get_category_metrics(informational_ids)
    
    # 计算tolerant类别的指标
    tolerant_ids = [i for i, class_name in id2class.items() if class_name in tolerant_classes]
    tolerant_p, tolerant_r, tolerant_f1, tolerant_support = get_category_metrics(tolerant_ids)
    
    # 输出高级类别的指标
    if accelerator.is_main_process:
        tqdm.write(f"\nHigh-level Category Metrics:")
        tqdm.write(f"Sensitive (support: {sensitive_support}): P={sensitive_p:.4f}, R={sensitive_r:.4f}, F1={sensitive_f1:.4f}")
        tqdm.write(f"Informational (support: {info_support}): P={info_p:.4f}, R={info_r:.4f}, F1={info_f1:.4f}")
        tqdm.write(f"Tolerant (support: {tolerant_support}): P={tolerant_p:.4f}, R={tolerant_r:.4f}, F1={tolerant_f1:.4f}")
    
    # 计算原始17个类别的总体指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted',
        zero_division=0,
        labels=label_keys
    )
    
    # Calculate per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0,
        labels=label_keys
    )
    
    
    # Log results using tqdm.write to avoid breaking progress bar
    if accelerator.is_main_process:
        eval_results = [
            f"\n{split} Results:",
            f"QPS: {qps:.2f}",
            f"Loss: {avg_loss:.4f}",
            f"Overall Precision: {precision:.4f}",
            f"Overall Recall: {recall:.4f}",
            f"Overall F1 Score: {f1:.4f}"
        ]
        
        # Add per-class metrics
        for i in range(len(label_keys)):
            eval_results.extend([
                f"\n{id2class[i]} Metrics:",
                f"Precision: {class_precision[i]:.4f}",
                f"Recall: {class_recall[i]:.4f}",
                f"F1 Score: {class_f1[i]:.4f}",
                f"Support: {class_support[i]}"
            ])
        
        # Output results to console
        tqdm.write('\n'.join(eval_results))
            
        # Log to tensorboard
        print(f'eval_results: eval_results')
        print(f'overall_precision: {precision}, overall_recall: {recall}, overall_f1: {f1}')
        
        # 记录原始类别的指标
        for i in range(len(label_keys)):
            class_name = id2class[i]
            print(f'{split}/{class_name}/precision: {class_precision[i]}')
            print(f'{split}/{class_name}/recall: {class_recall[i]}')
            print(f'{split}/{class_name}/f1: {class_f1[i]}')
            print(f'{split}/{class_name}/support: {class_support[i]}')
        
        # 记录整体指标
        
        # 记录三个高级类别的指标
        # Sensitive
        print(f'{split}/sensitive/precision: {sensitive_p}')
        print(f'{split}/sensitive/recall: {sensitive_r}')
        print(f'{split}/sensitive/f1: {sensitive_f1}')
        print(f'{split}/sensitive/support: {sensitive_support}')
        
        # Informational
        print(f'{split}/informational/precision: {info_p}')
        print(f'{split}/informational/recall: {info_r}')
        print(f'{split}/informational/f1: {info_f1}')
        print(f'{split}/informational/support: {info_support}')
        
        # Tolerant
        print(f'{split}/tolerant/precision: {tolerant_p}')
        print(f'{split}/tolerant/recall: {tolerant_r}')
        print(f'{split}/tolerant/f1: {tolerant_f1}')
        print(f'{split}/tolerant/support: {tolerant_support}')
        
    return avg_loss, precision, recall, f1
    
def load_model_and_tokenizer(model_path):
    """Load the best model and its tokenizer"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load model with float32 precision to avoid dtype mismatches
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32  # Force float32 precision
    )
    
    return model, tokenizer

def evaluate_test_set(args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Initialize accelerator with float32 precision
    accelerator = Accelerator(mixed_precision="no")  # Disable mixed precision
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    # test_df = pd.read_csv(args.test_file, sep='\t', header=None, names=['prompt', 'risk_type_label'])
    test_df = pd.read_csv(args.test_file, sep='\t')
    test_df = test_df[['prompt', 'risk_type_label']]

    # test_df = test_df.head(100)
        
    # Clean data
    test_df['prompt'] = test_df['prompt'].fillna('')
    
    # Create test dataset
    test_dataset = QueryDataset(
        test_df, 
        tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloader with consistent dtype
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(128, os.cpu_count()),
        pin_memory=True
    )
    
    # Prepare model and dataloader
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Evaluate
    logger.info("Starting evaluation...")
    # 评估模型
    metrics = evaluate(model, test_dataloader, accelerator)
    
    # 打印结果
    # logger.info("\nTest Set Results:")
    # logger.info(f"Loss: {metrics['loss']:.4f}")
    # logger.info(f"Overall Precision: {metrics['precision']:.4f}")
    # logger.info(f"Overall Recall: {metrics['recall']:.4f}")
    # logger.info(f"Overall F1: {metrics['f1']:.4f}")
    print(metrics)
   

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ModernBERT on test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test data file (tsv format)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Evaluation batch size")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_test_set(args)