#!/usr/bin/env python
import os
import warnings
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants from training script
LEVEL1_CLASSES = {
    0: 'Sensitive',
    1: 'Informational',
    2: 'Tolerant'
}

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
    'information': -100,
    'pirate media': 5,
    'adult': 4,
    'pirate information': 3,
    'pirate software': 1,
}

LEVEL2_REVERSE_MAPPING = {v: k for k, v in LEVEL2_MAPPING.items()}
SENSITIVE_VALUES = [100, 98, 96, 94, 90, 87, 85, 82, 80, 70, 60, 30]
TOLERANT_VALUES = [5, 4, 3, 1]

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

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, required_columns, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.required_columns = required_columns
        
        if isinstance(data, pd.DataFrame):
            data[required_columns[1]] = pd.to_numeric(data[required_columns[1]], errors='raise')
            data[required_columns[2]] = pd.to_numeric(data[required_columns[2]], errors='raise')
            self.dataset = Dataset.from_pandas(data)
        else:
            self.dataset = data
            
        self.process_data()
        
    def process_data(self):
        texts = self.dataset[self.required_columns[0]]
        self.encoded_data = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        self.level1_labels = torch.tensor(self.dataset[self.required_columns[1]], dtype=torch.long)
        self.level2_labels = torch.tensor(self.dataset[self.required_columns[2]], dtype=torch.long)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encoded_data['input_ids'][idx],
            'attention_mask': self.encoded_data['attention_mask'][idx],
            'labels': self.level1_labels[idx],
            'level2_labels': self.level2_labels[idx]
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_level1_preds = []
    all_level1_labels = []
    all_level1_probs = []
    all_sensitive_preds = []
    all_sensitive_labels = []
    all_tolerant_preds = []
    all_tolerant_labels = []
    
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['labels'].size(0)
            total_samples += batch_size
            
            outputs = model(**batch)
            
            # Level 1 predictions and probabilities
            level1_logits = outputs['level1_logits']
            level1_probs = torch.softmax(level1_logits, dim=-1)
            level1_preds = level1_probs.argmax(dim=-1)
            
            all_level1_preds.extend(level1_preds.cpu().numpy())
            all_level1_labels.extend(batch['labels'].cpu().numpy())
            all_level1_probs.extend(level1_probs.cpu().numpy())
            
            # Level 2 predictions
            sensitive_logits = outputs['sensitive_logits']
            tolerant_logits = outputs['tolerant_logits']
            
            # Process sensitive predictions
            sensitive_mask = (batch['labels'] == 0) & (batch['level2_labels'] != -100)
            if sensitive_mask.any():
                sensitive_preds = sensitive_logits[sensitive_mask].argmax(dim=-1)
                mapped_sensitive_preds = torch.zeros_like(sensitive_preds)
                for i, val in enumerate(SENSITIVE_VALUES):
                    mapped_sensitive_preds[sensitive_preds == i] = val
                all_sensitive_preds.extend(mapped_sensitive_preds.cpu().numpy())
                all_sensitive_labels.extend(batch['level2_labels'][sensitive_mask].cpu().numpy())
            
            # Process tolerant predictions
            tolerant_mask = (batch['labels'] == 2) & (batch['level2_labels'] != -100)
            if tolerant_mask.any():
                tolerant_preds = tolerant_logits[tolerant_mask].argmax(dim=-1)
                mapped_tolerant_preds = torch.zeros_like(tolerant_preds)
                for i, val in enumerate(TOLERANT_VALUES):
                    mapped_tolerant_preds[tolerant_preds == i] = val
                all_tolerant_preds.extend(mapped_tolerant_preds.cpu().numpy())
                all_tolerant_labels.extend(batch['level2_labels'][tolerant_mask].cpu().numpy())
    
    eval_time = time.time() - start_time
    qps = total_samples / eval_time
    
    # Calculate metrics
    level1_metrics = precision_recall_fscore_support(
        all_level1_labels,
        all_level1_preds,
        average='weighted',
        labels=list(LEVEL1_CLASSES.keys())
    )
    
    # Calculate AUPRC for level 1
    all_level1_probs = np.array(all_level1_probs)
    all_level1_labels = np.array(all_level1_labels)
    level1_auprc = {}
    for i in range(len(LEVEL1_CLASSES)):
        binary_labels = (all_level1_labels == i).astype(int)
        level1_auprc[LEVEL1_CLASSES[i]] = average_precision_score(binary_labels, all_level1_probs[:, i])
    
    results = {
        'qps': qps,
        'level1': {
            'precision': level1_metrics[0],
            'recall': level1_metrics[1],
            'f1': level1_metrics[2],
            'auprc': level1_auprc
        }
    }
    
    # Calculate level 2 metrics if available
    if all_sensitive_preds:
        sensitive_metrics = precision_recall_fscore_support(
            all_sensitive_labels,
            all_sensitive_preds,
            average='weighted',
            labels=SENSITIVE_VALUES
        )
        results['level2_sensitive'] = {
            'precision': sensitive_metrics[0],
            'recall': sensitive_metrics[1],
            'f1': sensitive_metrics[2]
        }
    
    if all_tolerant_preds:
        tolerant_metrics = precision_recall_fscore_support(
            all_tolerant_labels,
            all_tolerant_preds,
            average='weighted',
            labels=TOLERANT_VALUES
        )
        results['level2_tolerant'] = {
            'precision': tolerant_metrics[0],
            'recall': tolerant_metrics[1],
            'f1': tolerant_metrics[2]
        }
    
    return results

def load_and_evaluate(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model weights and config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Import HierarchicalClassifier
    from modernBERT_hierarchical_classifier import HierarchicalClassifier, AutoModel
    
    # Load base model
    base_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    model = HierarchicalClassifier(base_model, config)
    
    # Load the state dict from safetensors format
    from safetensors.torch import load_file
    
    # Load and merge sharded weights
    state_dict = {}
    index_file = os.path.join(args.model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        import json
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Load each shard
        for shard_file in index_data['weight_map'].values():
            shard_path = os.path.join(args.model_path, shard_file)
            shard_state = load_file(shard_path)
            state_dict.update(shard_state)
    else:
        # Try loading single safetensors file
        model_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
        else:
            raise FileNotFoundError(f"No model weights found in {args.model_path}")

    # Load state dict into model
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded successfully")
    
    print("Loading evaluation data...")
    required_columns = args.eval_columns.split(',')
    _, val_df, test_df = load_and_split_data(args.eval_file, required_columns)

    # Create dataset and dataloader
    eval_dataset = QueryDataset(test_df, tokenizer, required_columns, max_length=args.max_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, eval_dataloader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"QPS: {results['qps']:.2f}")
    
    print("\nLevel 1 Classification:")
    print(f"F1 Score: {results['level1']['f1']:.4f}")
    print(f"Precision: {results['level1']['precision']:.4f}")
    print(f"Recall: {results['level1']['recall']:.4f}")
    print("\nAUPRC per class:")
    for class_name, auprc in results['level1']['auprc'].items():
        print(f"{class_name}: {auprc:.4f}")
    
    if 'level2_sensitive' in results:
        print("\nLevel 2 - Sensitive Class:")
        print(f"F1 Score: {results['level2_sensitive']['f1']:.4f}")
        print(f"Precision: {results['level2_sensitive']['precision']:.4f}")
        print(f"Recall: {results['level2_sensitive']['recall']:.4f}")
    
    if 'level2_tolerant' in results:
        print("\nLevel 2 - Tolerant Class:")
        print(f"F1 Score: {results['level2_tolerant']['f1']:.4f}")
        print(f"Precision: {results['level2_tolerant']['precision']:.4f}")
        print(f"Recall: {results['level2_tolerant']['recall']:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ModernBERT hierarchical classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation data file (tsv format)")
    parser.add_argument("--eval_columns", type=str, required=True, help="Comma-separated list of columns: text,level1_label,level2_label")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    load_and_evaluate(args)
    
# python modernBERT_hierarchical_eval.py \
#     --model_path /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/hierarchical_classification/checkpoints/models_modernBERT_hierarchical_classifier/latest_model \
#     --eval_file /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all6_for_hierarchical1.tsv \ 
#     --eval_columns prompt,collapsed_label,level2_label \
#     --max_length 2048 \
#     --batch_size 128 \
#     --num_workers 4

# python modernBERT_hierarchical_eval.py --model_path /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/hierarchical_classification/checkpoints/models_modernBERT_hierarchical_classifier1/latest_model --eval_file /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all6_for_hierarchical1.tsv  --eval_columns prompt,collapsed_label,level2_label --max_length 2048 --batch_size 128 --num_workers 4
    