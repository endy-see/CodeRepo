#!/usr/bin/env python
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings
import logging
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from autogluon.multimodal import MultiModalPredictor
from datasets import Dataset

# 配置警告和日志
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo.utils").setLevel(logging.ERROR)

base_model_dir = '/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/AutoGluOn/HuggingFaceModels'

def evaluate(predictor, data, labels, compute_qps=False):
    start_time = time.time()
    predictions = predictor.predict(data)
    end_time = time.time()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    if compute_qps:
        total_time = end_time - start_time
        qps = len(data) / total_time if total_time > 0 else 0
        metrics["qps"] = float(qps)
        
    return metrics

def load_and_split_data(file_path, text_column, label_column, cache_dir=None):
    """Load data and split into train/val/test sets with 8:1:1 ratio"""
    if cache_dir and os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        dataset = Dataset.load_from_disk(cache_dir)
        df = dataset.to_pandas()
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path, sep='\t')
        
        if cache_dir:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            print(f"Caching dataset to {cache_dir}")
            dataset = Dataset.from_pandas(df)
            dataset.save_to_disk(cache_dir)
    
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Missing required columns: {text_column} or {label_column}")
    
    # Clean the data
    df[text_column] = df[text_column].fillna('')
    
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
    
    print(f'train data columns: {train_df.columns}')
    return train_df, val_df, test_df

def train_transformer_model(train_df, val_df, test_df, text_column, label_column, args, model_dir):
    """Train transformer-based model"""
    print(f"\nTraining Transformer model {args.checkpoint_name}...")
    
    predictor = MultiModalPredictor(
        label=label_column,
        problem_type='multiclass',
        eval_metric='f1_macro',
        validation_metric='f1_macro',
        num_classes=3,
        warn_if_exist=False,
        path=os.path.join(model_dir, f'{args.checkpoint_name}_collapsed_label'),
        enable_progress_bar=True,
    )
    
    hyperparameters = {
        'model.names': ['hf_text'],
        'model.hf_text.checkpoint_name': os.path.join(base_model_dir, args.checkpoint_name),
        'model.hf_text.max_text_len': args.max_length,
        'optimization.max_epochs': args.num_epochs,
        'optimization.learning_rate': args.learning_rate,        
        'optimization.top_k': 3,        
        'env.num_gpus': 4,
        # 'env.per_gpu_batch_size': args.batch_size,
        'env.batch_size': args.batch_size,
        'env.num_workers': 4,
    }
    
    # Train the model
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters=hyperparameters,
        seed=args.seed
    )
    
    # Evaluate after training
    val_metrics = evaluate(predictor, val_df[text_column], val_df[label_column], compute_qps=True)
    test_metrics = evaluate(predictor, test_df[text_column], test_df[label_column], compute_qps=True)
    
    # Print validation metrics
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  QPS: {val_metrics['qps']:.2f}")
    
    # Print test metrics
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  QPS: {test_metrics['qps']:.2f}")
    
    # Save final metrics and model info
    model_info = {
        'hyperparameters': predictor.get_hyperparameters(),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    model_info_path = os.path.join(model_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel training completed. Model info saved to {model_info_path}")
    
    return predictor

def model_training(args):
    # Load and split dataset
    text_column, label_column = args.train_columns.split(',')
    cache_dir = os.path.join(args.output_dir, f'cache_autogluon_{args.checkpoint_name}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_df, val_df, test_df = load_and_split_data(args.train_file, text_column, label_column, cache_dir)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Train transformer model
    predictor = train_transformer_model(
        train_df, val_df, test_df, text_column, label_column, args, args.output_dir
    )
    
    # Final evaluation
    final_metrics = evaluate(
        predictor, 
        test_df[text_column], 
        test_df[label_column]
    )
    print(f"\nFinal test metrics: {json.dumps(final_metrics, indent=2)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train AutoGluon text classifier")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to data file (tsv)")
    parser.add_argument("--train_columns", type=str, required=True,
                        help="Comma-separated list of text_column,label_column")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint_name", type=str, default='deberta-v3-large')
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_training(args)