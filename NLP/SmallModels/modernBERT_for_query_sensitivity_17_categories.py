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
from numpy import ndarray
import torch
import sys
import signal
from datasets import Dataset
import torch.distributed as dist
from torch.distributed import ReduceOp
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
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.tensorboard import SummaryWriter
import gc
import time
import torch._dynamo.config as dynamo_config
dynamo_config.capture_scalar_outputs = True

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

# id2class = {
#     0: 'Sensitive',
#     1: 'Informational',
#     2: "Tolerant"
# }
label_keys = list(id2class.keys())
        
# Add CUDA optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)

class QueryDataset(TorchDataset):
    """Custom dataset for Query and risk_type data with optimized tokenization"""
    def __init__(self, data, tokenizer, required_columns, max_length=512, cache_dir=None, name=None):
        self.name = name
        self.tokenizer = tokenizer
        self.required_columns = required_columns
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Convert data to Hugging Face dataset with memory optimization
        if isinstance(data, pd.DataFrame):
            # Free memory from pandas DataFrame after conversion
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
        return {
            'input_ids': self.encoded_data['input_ids'][idx],
            'attention_mask': self.encoded_data['attention_mask'][idx],
            'labels': self.labels[idx]
        }
            
    def load_from_cache(self):
        """Load tokenized data from cache directory with optimized batching"""
        # Get local rank for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.cache_dir:
            data_path = self.cache_dir / f'{self.name}_encoded_data_rank{local_rank}.pt'
            labels_path = self.cache_dir / f'{self.name}_labels_rank{local_rank}.pt'
            if data_path.exists() and labels_path.exists():
                from transformers.tokenization_utils_base import BatchEncoding
                from tokenizers import Encoding
                with torch.serialization.safe_globals([BatchEncoding, Encoding]):
                    self.encoded_data = torch.load(data_path, weights_only=False)
                    self.labels = torch.load(labels_path, weights_only=False)
                logger.info(f"Tokenized data loaded from {self.cache_dir} for rank {local_rank}")
                return
        
        logger.warning(f"Cache not found for name {self.name} and rank {local_rank}. Now tokenizing data...")
        # Pre-tokenize data in batches
        batch_size = 1000 # Process 1000 examples at a time
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
        
        # Convert labels to tensor
        self.labels = torch.tensor(self.dataset[self.required_columns[1]], dtype=torch.long)
        # self.save_to_cache()
        logger.info("Tokenized data saved to cache.")
        
        # Clear temporary lists
        del all_input_ids, all_attention_masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()    
    
    def save_to_cache(self):
        """Save tokenized data to cache directory"""
        if self.cache_dir:
            # Get local rank for distributed training
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoded_data, self.cache_dir / f'{self.name}_encoded_data_rank{local_rank}.pt')
            torch.save(self.labels, self.cache_dir / f'{self.name}_labels_rank{local_rank}.pt')
            logger.info(f"Tokenized data saved to {self.cache_dir} for rank {local_rank}")
        
    def clear_cache(self):
        """Clear cached data"""
        if self.cache_dir and self.cache_dir.exists():
            for file in self.cache_dir.glob('*'):
                file.unlink()
            logger.info(f"Cache cleared at {self.cache_dir}")
        else:
            logger.warning("Cache directory not found or already empty.")
            raise FileNotFoundError("Cache directory not found or already empty.")

def load_and_split_data(file_path, required_columns):
    """Load data and split into train/val/test sets with 8:1:1 ratio"""
    # Load the data
    df = pd.read_csv(file_path, sep='\t')
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
                # Save config and tokenizer first (small files)
                metadata = {
                    "num_labels": unwrapped_model.config.num_labels,
                    "id2label": id2class,
                    "label2id": {v:k for k, v in id2class.items()}
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
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # try:
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
    
    # Load tokenizer and model
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=args.num_labels, gradient_checkpointing=True)
    
    # Load and split dataset
    required_columns = args.train_columns.split(',')
    train_df, val_df, test_df = load_and_split_data(args.train_file, required_columns)
    
    # Create datasets
    train_dataset = QueryDataset(train_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='train')
    val_dataset = QueryDataset(val_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='val')
    test_dataset = QueryDataset(test_df, tokenizer, required_columns, max_length=args.max_length, cache_dir=args.cache_dir, name='test')
    
    # Create DataLoaders with optimized settings
    def create_dataloader(dataset, shuffle):
        # Calculate effective batch size considering all processes
        total_size = len(dataset)
        should_drop_last = shuffle  # Only drop last during training
        effective_batch_size = args.batch_size * accelerator.num_processes

        # Set sampler for distributed training
        if accelerator.use_distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=shuffle,
                drop_last=should_drop_last,
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
            drop_last=should_drop_last,
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
        
        # Calculate overall metrics
        avg_loss = total_loss.item() / len(dataloader)
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
        
        # Calculate high-level category metrics
        def get_category_metrics(class_ids):
            # Convert multi-class to binary for the category
            y_true_binary = np.isin(all_labels, class_ids).astype(int)
            y_pred_binary = np.isin(all_predictions, class_ids).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_binary,
                y_pred_binary,
                average='binary',
                zero_division=0
            )
            
            return precision, recall, f1
        
        # Get metrics for each high-level category
        sensitive_metrics = get_category_metrics(sensitive_ids)
        informational_metrics = get_category_metrics(informational_ids)
        tolerant_metrics = get_category_metrics(tolerant_ids)
        
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
            
            # Add high-level category metrics
            eval_results.extend([
                "\nHigh-level Category Metrics:",
                f"\nSensitive Metrics:",
                f"Precision: {sensitive_metrics[0]:.4f}",
                f"Recall: {sensitive_metrics[1]:.4f}",
                f"F1 Score: {sensitive_metrics[2]:.4f}",
                f"\nInformational Metrics:",
                f"Precision: {informational_metrics[0]:.4f}",
                f"Recall: {informational_metrics[1]:.4f}",
                f"F1 Score: {informational_metrics[2]:.4f}",
                f"\nTolerant Metrics:",
                f"Precision: {tolerant_metrics[0]:.4f}",
                f"Recall: {tolerant_metrics[1]:.4f}",
                f"F1 Score: {tolerant_metrics[2]:.4f}"
            ])
            
            # Use tqdm.write for all output
            tqdm.write('\n'.join(eval_results))
                
            # Log to tensorboard
            writer.add_scalar(f'{split}/loss', avg_loss, epoch)
            writer.add_scalar(f'{split}/overall_precision', precision, epoch)
            writer.add_scalar(f'{split}/overall_recall', recall, epoch)
            writer.add_scalar(f'{split}/overall_f1', f1, epoch)
            
            # Log per-class metrics to tensorboard
            for i in range(len(label_keys)):
                class_name = id2class[i]
                writer.add_scalar(f'{split}/{class_name}/precision', class_precision[i], epoch)
                writer.add_scalar(f'{split}/{class_name}/recall', class_recall[i], epoch)
                writer.add_scalar(f'{split}/{class_name}/f1', class_f1[i], epoch)
            
            # Log high-level category metrics
            categories = [
                ('Sensitive', sensitive_metrics),
                ('Informational', informational_metrics),
                ('Tolerant', tolerant_metrics)
            ]
            for cat_name, metrics in categories:
                writer.add_scalar(f'{split}/{cat_name}/precision', metrics[0], epoch)
                writer.add_scalar(f'{split}/{cat_name}/recall', metrics[1], epoch)
                writer.add_scalar(f'{split}/{cat_name}/f1', metrics[2], epoch)
        
        return avg_loss, precision, recall, f1
    
    # Calculate total steps considering gradient accumulation
    total_training_steps = len(train_dataloader)  # 4 GPUs
    progress_bar = tqdm(
        range(total_training_steps * args.num_epochs),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )
    
    # Training loop
    best_val_f1 = 0.0
    best_model_path = None
    
    for epoch in range(args.num_epochs):
        # Verify distributed training state at the start of each epoch
        accelerator.wait_for_everyone()
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed training state lost between epochs")
        
        if accelerator.is_main_process:
            tqdm.write(f"\nStarting epoch {epoch + 1}/{args.num_epochs}")
        
        model.train()
        total_loss = 0
        torch.cuda.empty_cache()
            
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass with autocast for mixed precision
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                total_loss += loss.detach().float()
                # Backward pass
                accelerator.backward(loss)
                # Advanced gradient clipping with adaptive threshold
                if accelerator.sync_gradients:
                    # Get current gradient norm
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Log large gradient norms for monitoring
                    if grad_norm > 1.0 and accelerator.is_main_process and step % 100 == 0:
                        tqdm.write(f"\nWarning: Large gradient norm detected: {grad_norm:.2f}")
                        
                    # Clear bad gradients if norm is too high
                    if grad_norm > 10.0:  # Reset if gradients are extremely large
                        optimizer.zero_grad()
                        tqdm.write(f"\nWarning: Gradient norm too large ({grad_norm:.2f}), skipping update")
                        continue
                # Optimizer and scheduler steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Synchronize processes
                if step % 10 == 0:  # Periodic synchronization
                    accelerator.wait_for_everyone()
            
            if accelerator.sync_gradients:
                # Update progress bar with more metrics (only on gradient sync steps)
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = total_loss.item() / (step + 1)
                progress_bar.update(args.gradient_accumulation_steps)
                progress_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}")   #  Step {step+1}/{(len(train_dataloader)//4)}  # 4 is GPU count
                # progress_bar.set_postfix(step=f'{step+1}/{(len(train_dataloader)//4)*args.num_epochs}' , lr=current_lr)
                # Log to tensorboard
                if accelerator.is_main_process:
                    global_step = epoch * len(train_dataloader) + step
                    writer.add_scalar('training/loss', avg_loss, global_step)
                    writer.add_scalar('training/learning_rate', current_lr, global_step)
            
        # Clear GPU cache after training epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
                
        # Evaluation after each epoch
        # Use tqdm.write instead of logger.info to avoid breaking progress bar
        if accelerator.is_main_process:
            tqdm.write(f"\nEvaluating epoch {epoch + 1}/{args.num_epochs}...")
        
        # Ensure all processes are synced before evaluation
        accelerator.wait_for_everyone()
        
        # Evaluate on validation set
        val_loss, val_precision, val_recall, val_f1 = evaluate(model, val_dataloader, "val")
        # Evaluate on test set
        test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, "test")
        
        # Save best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
            # Create checkpoint directory
            checkpoint_dir = Path(args.output_dir) / f"best_model_f1_{val_f1:.4f}_epoch_{epoch+1}"
            
            # Save model (this will handle distributed state internally)
            saved_path = save_model(model, tokenizer, checkpoint_dir, accelerator)
            
            if accelerator.is_main_process:
                # Log best scores to tensorboard
                writer.add_scalar('best/val_f1', val_f1, epoch)
                writer.add_scalar('best/test_f1', test_f1, epoch)
            
            # Also save latest model in a fixed location for easy loading
            latest_dir = Path(args.output_dir) / "latest_model"
            save_model(model, tokenizer, latest_dir, accelerator)
        
        # Make sure all processes are synced before next epoch
        accelerator.wait_for_everyone()           
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
    parser.add_argument("--num_labels", type=int, default=17,
                        help="Number of risk type categories (default: 3)")
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
