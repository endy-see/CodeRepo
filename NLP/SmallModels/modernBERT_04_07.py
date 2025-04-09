#!/usr/bin/env python
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

logger = get_logger(__name__)

class URLTextDataset(TorchDataset):
    """Custom dataset for URL and text data"""
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
        url = row['url']
        text = row['text']
        label = row['label']
        
        combined_text = f"{url} {text}"
        
        # Tokenize the text
        encoding = self.tokenizer(combined_text, 
                                    truncation=True, 
                                    padding='max_length', 
                                    max_length=self.max_length, 
                                    return_tensors='pt')
        
        # Create a dictionary to return
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }
        
        return item
    
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['url', 'text', 'label'])
    required_columns = ['url', 'text', 'label']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean the data
    df['text'] = df['text'].fillna('')
    df['url'] = df['url'].fillna('')
    
    return df

def training_function(args):
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info(accelerator.state)
    
    # Load and preprocess data
    logger.info("Loading dataset...")    
    # Initialize tokenizer and model
    logger.info("Loading tokenizer and model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=args.num_labels)
    
    # Load dataset
    train_df = load_data(args.train_file)
    train_dataset = URLTextDataset(train_df, tokenizer, max_length=args.max_length, cache_dir=args.cache_dir)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # Training loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    progress_bar = tqdm(
        range(args.num_epochs * len(train_dataloader)),
        disable=not accelerator.is_local_main_process
    )
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Epoch {epoch+1} - Average loss: {total_loss.item()/(step+1):.4f}"
                )
        
        # Save checkpoint at end of each epoch
        if accelerator.is_local_main_process:
            checkpoint_dir = Path(args.output_dir) / f"epoch_{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Unwrap and save model
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoint_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            
            # Save tokenizer
            if accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
                
            logger.info(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train ModernBERT classifier")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training data file (csv or tsv)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache tokenized data")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="Number of classification labels")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    training_function(args)
    
    
# /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/ModernBERT/data/TrainingSet.tsv
