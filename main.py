#!/usr/bin/env python3
import os
import glob
import json
import time
import signal
import pickle
from typing import List, Dict, Set
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from reformat_data import reformat_jsonl

# Global variables for graceful shutdown
SHOULD_STOP = False
TRAINER = None

def signal_handler(signum, frame):
    """Handle shutdown signals by setting global flag"""
    global SHOULD_STOP
    print("\nReceived shutdown signal. Will stop after current epoch...")
    SHOULD_STOP = True

def load_processed_files(tracking_file: str) -> Set[str]:
    """Load the set of already processed files"""
    if os.path.exists(tracking_file):
        with open(tracking_file, 'rb') as f:
            return pickle.load(f)
    return set()

def save_processed_files(tracking_file: str, processed_files: Set[str]):
    """Save the set of processed files"""
    with open(tracking_file, 'wb') as f:
        pickle.dump(processed_files, f)

def load_and_process_data(worker_dirs: List[str], processed_files_path: str) -> List[Dict]:
    """Load and process data from multiple worker directories, tracking processed files"""
    all_data = []
    processed_files = load_processed_files(processed_files_path)
    newly_processed = set()
    
    for worker_dir in worker_dirs:
        json_files = glob.glob(os.path.join(worker_dir, "*.json"))
        
        for json_file in json_files:
            abs_path = os.path.abspath(json_file)
            if abs_path in processed_files:
                print(f"Loading previously processed file: {json_file}")
            else:
                print(f"Processing new file: {json_file}")
                reformat_jsonl(json_file)
                newly_processed.add(abs_path)
            
            # Read the file regardless of whether it's been processed before
            with open(json_file, 'r') as f:
                data = json.loads(f.read().strip())
                all_data.append(data)
    
    # Update processed files with any new ones
    if newly_processed:
        processed_files.update(newly_processed)
        save_processed_files(processed_files_path, processed_files)
    
    return all_data

def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Convert data into a format suitable for training."""
    formatted_data = []
    
    for item in data:
        if "messages" in item:
            # Convert messages to conversation format
            conversation = ""
            for msg in item["messages"]:
                role = msg["role"]
                content = msg.get("content", "")
                
                if role == "user":
                    conversation += f"<|user|>{content}</s>"
                elif role == "assistant":
                    conversation += f"<|assistant|>{content}</s>"
                elif role == "tool":
                    conversation += f"<|tool|>{content}</s>"
            
            formatted_data.append({"text": conversation})
    
    # Create initial dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        batched=True
    )
    
    print(f"Created dataset with {len(formatted_data)} samples")
    
    return tokenized_dataset

def setup_model_and_tokenizer():
    """Initialize and configure the model and tokenizer."""
    # Quantization config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Add special tokens
    special_tokens = ["<|user|>", "<|assistant|>", "<|tool|>", "</s>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters."""
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

class TimedTrainer(Trainer):
    """Custom trainer that checks time limits during training"""
    def __init__(self, *args, **kwargs):
        self.start_time = time.time()
        self.time_limit = kwargs.pop('time_limit', 3060)  # 51 minutes in seconds
        super().__init__(*args, **kwargs)
    
    def training_step(self, *args, **kwargs):
        """Check time limit before each training step"""
        global SHOULD_STOP
        if time.time() - self.start_time > self.time_limit or SHOULD_STOP:
            print("\nReached time limit or received stop signal. Stopping training...")
            raise KeyboardInterrupt
        return super().training_step(*args, **kwargs)

def main():
    global TRAINER
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Define worker directories
    worker_dirs = [
        "../output/worker1",
        "../output/worker2",
        "../output/worker3"
    ]
    
    # Setup output directories in scratch space
    scratch_dir = os.path.join("/data/scratch", os.environ["USER"], "mira_finetune")
    results_dir = os.path.join(scratch_dir, "results")
    final_model_dir = os.path.join(scratch_dir, "final_model")
    processed_files_path = os.path.join(scratch_dir, "processed_files.pkl")
    
    # Create directories if they don't exist
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    data = load_and_process_data(worker_dirs, processed_files_path)
    
    if not data:
        print("No data files found. Exiting...")
        return
    
    print(f"Loaded {len(data)} files for training")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    print("Configuring LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(data, tokenizer)
    
    # Calculate steps for one complete pass through dataset
    total_samples = len(dataset)
    batch_size = 4  # per_device_train_batch_size
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = total_samples // effective_batch_size
    total_steps = steps_per_epoch * 1000  # 1000 epochs
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=1000,  # 1000 epochs
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=steps_per_epoch,  # Log once per epoch
        save_strategy="epoch",
        save_steps=steps_per_epoch,  # Save every epoch
        warmup_ratio=0.01,  # Shorter warmup for many epochs
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        # Load from checkpoint if available
        resume_from_checkpoint=True,
        # Additional settings for long training
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=0.3,  # Help stability in long training
        weight_decay=0.01,  # Help prevent overfitting
        lr_scheduler_type="cosine_with_restarts",  # Better for many epochs
        num_cycles=50,  # Restart learning rate ~every 20 epochs
    )
    
    print(f"Training for 1000 epochs")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Saving checkpoints every epoch ({steps_per_epoch} steps)")
    print(f"Learning rate will cycle {training_args.num_cycles} times")
    
    # Initialize trainer with time limit
    TRAINER = TimedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda data: {'input_ids': torch.stack([x['input_ids'] for x in data]),
                                  'attention_mask': torch.stack([x['attention_mask'] for x in data])},
        time_limit=3060  # 51 minutes
    )
    
    try:
        # Start training
        print("Starting training...")
        TRAINER.train()
        
        # Save the final model only if training completed successfully
        print("Saving model...")
        TRAINER.save_model(final_model_dir)
        
        # Create symlinks in the current directory for easy access
        current_dir = os.getcwd()
        for link_name, target_dir in [("results", results_dir), ("final_model", final_model_dir)]:
            link_path = os.path.join(current_dir, link_name)
            if not os.path.exists(link_path):
                os.symlink(target_dir, link_path)
        
        print(f"Training completed. Model saved in {final_model_dir}")
        print(f"Training results and checkpoints saved in {results_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        TRAINER.save_model(os.path.join(results_dir, "checkpoint-interrupted"))
        print(f"Checkpoint saved in {results_dir}/checkpoint-interrupted")
    
if __name__ == "__main__":
    main()
