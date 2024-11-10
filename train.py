import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import tqdm as notebook_tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup

batch_size = 5
max_seq_len = 50
num_epochs = 1

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        initial = self.data.iloc[index]["init"]
        target = self.data.iloc[index]["target"]
        
        # Specifying prompt 
        prompt = "Reformulate the query: "
        initial = prompt + initial
        
        initial = self.tokenizer(initial, padding="max_length", max_length=max_seq_len, 
                                          truncation=True, return_tensors='pt').input_ids
        target = self.tokenizer(target, padding="max_length", max_length=max_seq_len, 
                                          truncation=True, return_tensors='pt').input_ids
        # each index in initial correponds to its input text string, and size of max_seq_len 
        # squeeze to remove extra dimension
        return {"input": initial.squeeze(), "target": target.squeeze()}

def train(model, dataloader, optimizer, scheduler, device, num_epochs):
    model.train()
    total_loss = 0
    step = 0
    for batch in dataloader:
        input_ids = batch['input'].to(device)
        target_ids = batch['target'].to(device)
        labels = target_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        step += 1
        if (step%1000==0): print(f"At step {step}, the total_loss is {total_loss}")
        
        loss.backward()
        optimizer.step()  
        scheduler.step() 
        optimizer.zero_grad() 

    avg_train_loss = total_loss / len(dataloader)
    avg_test_loss = evaluate(model, test_data, device)
    print(f"avg training loss: {avg_train_loss}, avg test loss: {avg_test_loss}")
    torch.save({
        'model_state_dict': model.state_dict()
    }, f"test_model_prompt.pt")    

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input'].to(device)
            target_ids = batch['target'].to(device)
            labels = target_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss



if __name__ == '__main__':
    raw_data = pd.read_csv("diamond.tsv", sep="\t", header=None, 
                       names=["qid","init","map_init","target","map_target"])
    raw_data = raw_data[["init","target"]]
    cutoff = int(0.9*len(raw_data))
    train_data = CustomDataset(raw_data[:cutoff], tokenizer)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CustomDataset(raw_data[cutoff:], tokenizer)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_epochs*len(train_data))

    for epoch in range(num_epochs):
        train(model, train_data, optimizer, scheduler, device, epoch)
