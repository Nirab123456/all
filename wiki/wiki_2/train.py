from transformers import BertForMaskedLM, BertTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# Load your text data as a LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='C:/langchain2/wiki/wiki_2/trext_data/processed_text.txt',
    block_size=128,
)

# Prepare the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.0000001
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='C:/langchain2/wiki/wiki_2/trext_data/roberta_model_1',
    num_train_epochs=500,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_steps=100,
    logging_dir='./logs',
    overwrite_output_dir=True,
)

# Create the trainer and start pretraining
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

