# Load the fine-tuned XLM-RoBERTa model
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
model = XLMRobertaForSequenceClassification.from_pretrained('C:/langchain2/wiki/wiki_2/trext_data/roberta_model_1')
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

# Define the question to ask
question = "What is the cause of dengue fever?"

# Define the possible answers to the question
answers = [
    "Dengue is related with dance.",
    "Dengue is related with water.",
    "Dengue is related with game."
]

# Loop through the answers and make predictions
scores = []
index = None
for i, answer in enumerate(answers):
    # Convert the question and answer to input features
    inputs = tokenizer.encode_plus(question, answer, return_tensors='pt')
    
    # Make the prediction
    with torch.no_grad():
        logits = model(**inputs)[0]
        prediction = torch.argmax(logits, dim=1)
        score = torch.softmax(logits, dim=1).mean().item()
        
    # Update the index if a higher score is found
    if index is None or score > scores[index]:
        index = i
        
    scores.append(score)

# Print the answer with the highest score
if index is not None:
    print(f"The answer is: {answers[index]}")
else:
    print("No answers found.") 
