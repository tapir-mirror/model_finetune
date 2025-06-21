from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    output_dir="./mistral-finetuned"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()
