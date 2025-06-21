#!/usr/bin/env python3
import os
import glob
import json
from typing import List, Dict
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

def load_and_process_data(worker_dirs: List[str]) -> List[Dict]:
    """Load and process data from multiple worker directories."""
    all_data = []
    
    for worker_dir in worker_dirs:
        json_files = glob.glob(os.path.join(worker_dir, "*.json"))
        
        # Process each file through reformat_data
        for json_file in json_files:
            print(f"Processing {json_file}")
            reformat_jsonl(json_file)
            
            # Read the reformatted file
            with open(json_file, 'r') as f:
                data = json.loads(f.read().strip())
                all_data.append(data)
    
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
    
    return Dataset.from_list(formatted_data)

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

def main():
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
    
    # Create directories if they don't exist
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Load and process data
    print("Loading and processing data...")
    data = load_and_process_data(worker_dirs)
    
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=lambda data: {'input_ids': torch.stack([x['input_ids'] for x in data]),
                                  'attention_mask': torch.stack([x['attention_mask'] for x in data])}
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model(final_model_dir)
    
    # Create a symlink in the current directory for easy access
    current_dir = os.getcwd()
    os.symlink(results_dir, os.path.join(current_dir, "results"))
    os.symlink(final_model_dir, os.path.join(current_dir, "final_model"))
    
    print(f"Training completed. Model saved in {final_model_dir}")
    print(f"Training results and checkpoints saved in {results_dir}")
    
if __name__ == "__main__":
    main()
