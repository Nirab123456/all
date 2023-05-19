# Load the fine-tuned XLM-RoBERTa model
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

model_path = 'C:/langchain2/wiki/wiki_2/trext_data/roberta_model_1/checkpoint-500'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained(model_path)

# Define the question to ask
question = "What is the cause of dengue?"
# Define the answer
answer = "virus."

# Convert the question and answer to input features
inputs = tokenizer.encode_plus(question, answer, return_tensors='pt')

# Make the prediction
with torch.no_grad():
    logits = model(**inputs)[0]
    prediction = torch.argmax(logits, dim=1)
    score = torch.softmax(logits, dim=1).mean().item()

# Print the answer and score
print(f"The answer is: {answer}")
print(f"The score is: {score}")
