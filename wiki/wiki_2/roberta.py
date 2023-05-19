
from pathlib import Path
import torch
from transformers import BertTokenizer
from transformers import AutoModelForMaskedLM
import torch 
from transformers import AdamW
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

torch.cuda.empty_cache()
paths = [str(x) for x in Path('C:/langchain2/wiki/wiki_2/trext_data').glob('**/*.txt')]
# initialize an empty string to hold all the text
text = ""

# loop over the file paths and read each file into the text string
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        text += f.read()
text = text.replace('\n', ' ')
# split the text into smaller chunks
chunks = [text[i:i+100] for i in range(0, len(text), 100)]
#tokenize the chunks
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# create features for each chunk
features = []
for chunk in chunks:
    encoding = tokenizer.encode_plus(
        chunk,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    features.append(encoding)

# stack the features
input_ids = torch.cat([f['input_ids'] for f in features], dim=0)
attention_mask = torch.cat([f['attention_mask'] for f in features], dim=0)
token_type_ids = torch.cat([f['token_type_ids'] for f in features], dim=0)
#embeddings 
encodings = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': token_type_ids
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self,i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}
    
dataset = Dataset(encodings)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)           
# max_bytes=1024*1024*1024

model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# device = torch.device('cpu')
device = torch.device('cudanvdia -msi')

model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
from torch.cuda.amp import autocast, GradScaler
batch_size=10000
epoch = 4
scaler = GradScaler()
loop = tqdm(dataloader, leave=True)


for i, batch in enumerate(loop):
    if i % batch_size == 0:
        optim.zero_grad()

    with autocast():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    scaler.scale(loss).backward(retain_graph=True)

    if (i + 1) % batch_size == 0:
        scaler.step(optim)
        scaler.update()

        loop.set_description(f'Epoch:{epoch}')
        loop.set_description(f'loss:{loss.item()}')

    # Free up GPU memory after every batch
    del input_ids, attention_mask, labels, outputs
    torch.cuda.empty_cache()


model.save_pretrained('C:/langchain2/wiki/wiki_2/trext_data/roberta_model_1')
