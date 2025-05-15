import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import charactertokenizer
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

# --- Config ---
model_dir = 'ai-forever/charllama-1.3B'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

batch_size = 8
num_epochs = 1
learning_rate = 5e-5
max_length = 256
gradient_accumulation_steps = 4
save_steps = 1000
log_interval = 10  # Log every 10 iterations
output_dir = "charllama-finetuned"  # Directory to save checkpoints
log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)

# --- Dataset class ---
class CharDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self._load_and_preprocess(file_path)

    def _load_and_preprocess(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        return [line.strip().upper() for line in lines if len(line.strip()) > 10]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenizer(sentence,
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- Load tokenizer ---
tokenizer = charactertokenizer.CharacterTokenizer.from_pretrained(model_dir)

# --- Load model, optimizer, scheduler, and scaler from checkpoint (if available) ---
checkpoint_path = None  # Set to the checkpoint directory if you want to resume training
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Will be updated if loading from checkpoint
    num_training_steps=1 # Initialize with a non-zero value to avoid the check if loading
)
scaler = torch.cuda.amp.GradScaler()
global_step = 0
start_epoch = 0

if os.path.exists(output_dir):
    checkpoint_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and "checkpoint-step-" in d]
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        checkpoint_path = latest_checkpoint
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'scheduler.pt')))
        scaler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'scaler.pt')))
        global_step = int(checkpoint_path.split('-')[-1])
        train_dataset_temp = CharDataset("/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train.wrd", tokenizer, max_length)
        start_epoch = global_step // len(DataLoader(train_dataset_temp, batch_size=batch_size))
        print(f"Resuming from global step: {global_step}, epoch: {start_epoch}")
    else:
        print("No checkpoints found. Starting training from scratch.")
        train_dataset_temp = CharDataset("/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train.wrd", tokenizer, max_length)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(DataLoader(train_dataset_temp, batch_size=batch_size)) // 10,
            num_training_steps=len(DataLoader(train_dataset_temp, batch_size=batch_size)) * num_epochs // gradient_accumulation_steps
        )
else:
    os.makedirs(output_dir, exist_ok=True)
    print("Starting training from scratch.")
    train_dataset_temp = CharDataset("/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train.wrd", tokenizer, max_length)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(DataLoader(train_dataset_temp, batch_size=batch_size)) // 10,
        num_training_steps=len(DataLoader(train_dataset_temp, batch_size=batch_size)) * num_epochs // gradient_accumulation_steps
    )


model = torch.compile(model)
model.train()


# --- Create dataset and dataloader ---
train_file_path = "/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train_norm.txt"
dataset = CharDataset(train_file_path, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

# --- Training Loop ---
model.zero_grad()
for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), initial=global_step % len(dataloader) if start_epoch == epoch else 0, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for step, batch in progress_bar:
        if start_epoch == epoch and step < global_step % len(dataloader):
            continue  # Skip steps already done in the previous run

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            )
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Log loss and learning rate to TensorBoard every log_interval steps
            if global_step % log_interval == 0:
                writer.add_scalar('loss/step', loss.item() * gradient_accumulation_steps, global_step)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step)

            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
                torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, 'scaler.pt'))
                print(f"Checkpoint saved at step {global_step} to {checkpoint_dir}")

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}", "step": global_step + 1})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    writer.add_scalar('loss/epoch', avg_loss, epoch + 1)

# --- Save the final trained model ---
final_output_dir = os.path.join(output_dir, "final-model")
os.makedirs(final_output_dir, exist_ok=True)
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"Final model saved to {final_output_dir}")

# --- Close TensorBoard writer ---
writer.close()
print(f"TensorBoard logs saved to {log_dir}")
print("To view TensorBoard logs, run: `tensorboard --logdir runs` from your terminal.")