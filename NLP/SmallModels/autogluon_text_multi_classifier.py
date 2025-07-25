#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from sklearn.metrics import precision_recall_fscore_support
from autogluon.multimodal import MultiModalPredictor
from datasets import Dataset

def evaluate(predictor, data, labels):
    """Evaluate model predictions and return metrics"""
    predictions = predictor.predict(data)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
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

def load_and_split_data(file_path, text_column, label_column, cache_dir=None):
    """Load data and split into train/val/test sets with 8:1:1 ratio"""
    if cache_dir and os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        dataset = Dataset.load_from_disk(cache_dir)
        df = dataset.to_pandas()
    else:
        # Load the data
        df = pd.read_csv(file_path, sep='\t')
        
        if cache_dir:
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
    
    return train_df, val_df, test_df

def train_transformer_model(train_df, val_df, test_df, text_column, label_column, args, model_dir):
    """Train transformer-based model"""
    print("\nTraining Transformer model (DeBERTa-v3)...")
    
    predictor = MultiModalPredictor(
        label=label_column,
        problem_type='multiclass',
        eval_metric='f1',
        path=os.path.join(model_dir, 'transformer')
    )
    
    hyperparameters = {
        'model.names': ['hf_text'],
        'model.hf_text.checkpoint_name': 'microsoft/deberta-v3-base',
        'env.num_gpus': 4,
        'env.per_gpu_batch_size': args.batch_size,
        'env.num_workers': 4,
        'env.pin_memory': True,
        'env.gradient_accumulation_steps': 4,
        'env.distributed_strategy': 'ddp',
        'env.mixed_precision': 'fp16',
        
        'optimization.learning_rate': args.learning_rate,
        'optimization.max_epochs': args.num_epochs,
        'optimization.warmup_ratio': 0.1,
        'optimization.patience': 3,
        'optimization.optimizer.weight_decay': 0.01,
        'optimization.lr_scheduler': 'cosine',
        'optimization.lr_scheduler_kwargs': {'num_cycles': 0.5},
        'optimization.early_stopping': True,
        'optimization.early_stopping_patience': 5,
        'optimization.early_stopping_monitor': 'val_f1',
        'optimization.early_stopping_mode': 'max',
        'optimization.learning_rate_decay_factor': 0.1,
        'optimization.min_learning_rate': 1e-6,
        
        'model.hf_text.efficient_finetune': 'lora',
        'model.hf_text.adapter_dim': 64,
        'model.hf_text.max_text_len': args.max_length,
        'model.hf_text.gradient_checkpointing': True,
        'model.hf_text.insert_sep': True,
        'model.hf_text.pooling_mode': 'cls',
        'model.hf_text.mixed_precision': True,
        'model.hf_text.optimize_memory_usage': True,
        'model.hf_text.attention_probs_dropout_prob': 0.1,
        'model.hf_text.hidden_dropout_prob': 0.1,
        'model.hf_text.layer_norm_eps': 1e-7
    }
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters=hyperparameters,
        seed=args.seed
    )
    
    return predictor

def train_tree_models(train_df, val_df, test_df, text_column, label_column, args, model_dir):
    """Train tree-based models"""
    print("\nTraining Tree-based models (LightGBM, XGBoost, CatBoost)...")
    
    predictor = MultiModalPredictor(
        label=label_column,
        problem_type='multiclass',
        eval_metric='f1',
        path=os.path.join(model_dir, 'trees')
    )
    
    hyperparameters = {
        'model.names': ['numerical_transformer', 'categorical_transformer', 'fusion'],
        'model.numerical_transformer.presets': 'medium_quality',
        'model.categorical_transformer.presets': 'medium_quality',
        'model.fusion.presets': 'medium_quality',
        'env.num_gpus': 4,
        'env.per_gpu_batch_size': args.batch_size,
        'env.num_workers': 4,
        'env.pin_memory': True,
        
        'optimization.learning_rate': args.learning_rate,
        'optimization.max_epochs': args.num_epochs,
        'optimization.warmup_ratio': 0.1,
        'optimization.patience': 3,
        'optimization.early_stopping': True,
        'optimization.early_stopping_patience': 5,
        'optimization.early_stopping_monitor': 'val_f1',
        'optimization.early_stopping_mode': 'max',
        
        'preprocessing.text.ngram_range': (1, 5),
        'preprocessing.text.max_features': 20000,
        'preprocessing.text.use_tfidf': True,
        'preprocessing.text.use_fasttext': True,
        'preprocessing.text.fasttext_model': 'wiki.simple',
        'preprocessing.text.clean_text': True,
        'preprocessing.text.lower_case': True,
        'preprocessing.text.remove_html': True,
        'preprocessing.text.remove_special_chars': True,
        
        'model.numerical_transformer.extra_trees': True,
        'model.numerical_transformer.n_estimators': 200,
        'model.numerical_transformer.max_depth': 8,
        'model.categorical_transformer.cat_features_auto': True
    }
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters=hyperparameters,
        seed=args.seed
    )
    
    return predictor

def train_stacking_model(train_df, val_df, test_df, text_column, label_column, args, model_dir):
    """Train stacking model combining transformers and trees"""
    print("\nTraining Stacking model (Transformer + Trees)...")
    
    predictor = MultiModalPredictor(
        label=label_column,
        problem_type='multiclass',
        eval_metric='f1',
        path=os.path.join(model_dir, 'stacking')
    )
    
    hyperparameters = {
        'model.names': ['hf_text', 'numerical_transformer', 'categorical_transformer', 'fusion'],
        'model.hf_text.checkpoint_name': 'microsoft/deberta-v3-base',
        'model.numerical_transformer.presets': 'medium_quality',
        'model.categorical_transformer.presets': 'medium_quality',
        'model.fusion.presets': 'medium_quality',
        
        'env.num_gpus': 4,
        'env.per_gpu_batch_size': args.batch_size,
        'env.num_workers': 4,
        'env.pin_memory': True,
        'env.gradient_accumulation_steps': 4,
        'env.distributed_strategy': 'ddp',
        'env.mixed_precision': 'fp16',
        
        'optimization.learning_rate': args.learning_rate,
        'optimization.max_epochs': args.num_epochs,
        'optimization.warmup_ratio': 0.1,
        'optimization.patience': 3,
        'optimization.optimizer.weight_decay': 0.01,
        'optimization.lr_scheduler': 'cosine',
        'optimization.lr_scheduler_kwargs': {'num_cycles': 0.5},
        'optimization.early_stopping': True,
        'optimization.early_stopping_patience': 5,
        'optimization.early_stopping_monitor': 'val_f1',
        'optimization.early_stopping_mode': 'max',
        'optimization.learning_rate_decay_factor': 0.1,
        'optimization.min_learning_rate': 1e-6,
        
        'model.hf_text.efficient_finetune': 'lora',
        'model.hf_text.adapter_dim': 64,
        'model.hf_text.max_text_len': args.max_length,
        'model.hf_text.gradient_checkpointing': True,
        'model.hf_text.insert_sep': True,
        'model.hf_text.pooling_mode': 'cls',
        'model.hf_text.mixed_precision': True,
        'model.hf_text.optimize_memory_usage': True,
        'model.hf_text.attention_probs_dropout_prob': 0.1,
        'model.hf_text.hidden_dropout_prob': 0.1,
        'model.hf_text.layer_norm_eps': 1e-7,
        
        'preprocessing.text.ngram_range': (1, 5),
        'preprocessing.text.max_features': 20000,
        'preprocessing.text.use_tfidf': True,
        'preprocessing.text.use_fasttext': True,
        'preprocessing.text.fasttext_model': 'wiki.simple',
        'preprocessing.text.clean_text': True,
        'preprocessing.text.lower_case': True,
        'preprocessing.text.remove_html': True,
        'preprocessing.text.remove_special_chars': True,
        
        'model.ensemble_weights': {
            'hf_text': 0.5,
            'numerical_transformer': 0.2,
            'categorical_transformer': 0.2,
            'fusion': 0.1
        }
    }
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters=hyperparameters,
        seed=args.seed
    )
    
    return predictor

def train_all_models(args):
    # Load and split dataset
    text_column, label_column = args.train_columns.split(',')
    cache_dir = os.path.join(args.output_dir, 'cache_autogluon')
    train_df, val_df, test_df = load_and_split_data(args.train_file, text_column, label_column, cache_dir)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    models = {}
    metrics = {}
    
    # Train transformer model
    models['transformer'] = train_transformer_model(
        train_df, val_df, test_df, text_column, label_column, args, args.output_dir
    )
    metrics['transformer'] = evaluate(models['transformer'], test_df[text_column], test_df[label_column])
    
    # Train tree models
    models['trees'] = train_tree_models(
        train_df, val_df, test_df, text_column, label_column, args, args.output_dir
    )
    metrics['trees'] = evaluate(models['trees'], test_df[text_column], test_df[label_column])
    
    # Train stacking model
    models['stacking'] = train_stacking_model(
        train_df, val_df, test_df, text_column, label_column, args, args.output_dir
    )
    metrics['stacking'] = evaluate(models['stacking'], test_df[text_column], test_df[label_column])
    
    # Print comparison results
    print("\nModel Comparison Results on Test Set:")
    print("=====================================")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"Macro Precision: {model_metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {model_metrics['macro_recall']:.4f}")
        print(f"Macro F1: {model_metrics['macro_f1']:.4f}")
    
    # Find best model
    best_model = max(metrics.items(), key=lambda x: x[1]['macro_f1'])
    print(f"\nBest model: {best_model[0].upper()} with F1 score: {best_model[1]['macro_f1']:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, best_model[0])}")
    
    return models, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train AutoGluon text classifier")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to data file (tsv)")
    parser.add_argument("--train_columns", type=str, required=True,
                        help="Comma-separated list of text_column,label_column")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_all_models(args)