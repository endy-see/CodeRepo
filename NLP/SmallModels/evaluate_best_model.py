#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from sklearn.metrics import precision_recall_fscore_support
import logging
import argparse

def setup_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

logger = setup_logger()

class QueryDataset(TorchDataset):
    """Custom dataset for Query and risk_type data"""
    def __init__(self, data, tokenizer, max_length=512, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Convert data to Hugging Face dataset for better memory efficiency
        if isinstance(data, pd.DataFrame):
            self.dataset = Dataset.from_pandas(data)
        else:
            self.dataset = data
            
        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]
        query = row['query']
        label = row['risk_type']
        
        # Tokenize the query
        encoding = self.tokenizer(query,
                               truncation=True,
                               padding='max_length',
                               max_length=self.max_length,
                               return_tensors='pt')
        
        # Create a dictionary to return
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        return item

def load_and_split_data(file_path):
    """Load data and split into train/val/test sets with 8:1:1 ratio"""
    # Load the data
    df = pd.read_csv(file_path, sep='\t')
    required_columns = ['query', 'risk_type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean the data
    df['query'] = df['query'].fillna('')
    
    # Convert risk_type to numeric labels if needed
    if df['risk_type'].dtype == 'object':
        risk_type_map = {cat: idx for idx, cat in enumerate(df['risk_type'].unique())}
        df['risk_type'] = df['risk_type'].map(risk_type_map)
    
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

def evaluate(model, dataloader, accelerator):
    """Evaluate model on dataloader and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics for each class with zero_division=0
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist()
    }
    
    return metrics

def evaluate_model(args):
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Load data
    logger.info("Loading and splitting data...")
    _, val_df, test_df = load_and_split_data(args.data_file)
    
    # Load best model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create datasets
    val_dataset = QueryDataset(val_df, tokenizer, max_length=args.max_length)
    test_dataset = QueryDataset(test_df, tokenizer, max_length=args.max_length)
    
    # Create dataloaders
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Prepare for distributed evaluation
    model, val_dataloader, test_dataloader = accelerator.prepare(
        model, val_dataloader, test_dataloader
    )
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_dataloader, accelerator)
    
    logger.info("Validation Metrics:")
    logger.info(f"Macro Precision: {val_metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {val_metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {val_metrics['macro_f1']:.4f}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_dataloader, accelerator)
    
    logger.info("Test Metrics:")
    logger.info(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Print per-class metrics for both sets
    logger.info("\nValidation Per-class metrics:")
    for i in range(len(val_metrics['per_class_f1'])):
        logger.info(f"Class {i}:")
        logger.info(f"  Precision: {val_metrics['per_class_precision'][i]:.4f}")
        logger.info(f"  Recall: {val_metrics['per_class_recall'][i]:.4f}")
        logger.info(f"  F1: {val_metrics['per_class_f1'][i]:.4f}")
    
    logger.info("\nTest Per-class metrics:")
    for i in range(len(test_metrics['per_class_f1'])):
        logger.info(f"Class {i}:")
        logger.info(f"  Precision: {test_metrics['per_class_precision'][i]:.4f}")
        logger.info(f"  Recall: {test_metrics['per_class_recall'][i]:.4f}")
        logger.info(f"  F1: {test_metrics['per_class_f1'][i]:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best model on validation and test sets")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the best model directory")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to data file with 'query' and 'risk_type' columns (tsv)")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)