import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from baisline import BudgetTrainer, ComputeBudget, FlopBudget, HuggingFaceModelTrainer, PytorchFlopProfiler
import os

class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer, seq_len=128, max_samples=None):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if max_samples:
            text = text[:max_samples * seq_len * 4]
            
        self.encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=seq_len * max_samples if max_samples else None)
        
        self.input_ids = self.encodings.input_ids[0]
        self.seq_len = seq_len
        self.num_samples = max(1, self.input_ids.size(0) // seq_len)
        
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)

    def __len__(self): 
        return self.num_samples
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        chunk = self.input_ids[start:end]
        if chunk.size(0) < self.seq_len:
             padding = torch.full((self.seq_len - chunk.size(0),), tokenizer.eos_token_id, dtype=torch.long)
             chunk = torch.cat([chunk, padding])
             
        return {"input_ids": chunk, "labels": chunk.clone()}

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.eos_token_id or 0
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name)

text_file = "tiny_shakespeare.txt"
if not os.path.exists(text_file):
    import urllib.request
    print("Downloading dataset...")
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', text_file)

train_dataset = TextDataset(text_file, tokenizer, seq_len=64, max_samples=10)
train_loader = DataLoader(train_dataset, batch_size=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

budget = ComputeBudget(flop_budget=FlopBudget(max_flops=1e11)) 

trainer = BudgetTrainer(
    model=model,
    train_dataloader=train_loader,
    optimizer=optimizer,
    budget=budget,
    model_trainer=HuggingFaceModelTrainer(),
    model_name="GPT-2"
)

if __name__ == "__main__":
    print("Starting HF Transformer training...")
    results = trainer.fit()
    print("Training finished.")
    results.export_report("testcases/out/hf_gpt2_report.json")
