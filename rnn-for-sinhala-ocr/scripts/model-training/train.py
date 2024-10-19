from torch.utils.data import DataLoader
from datasets import load_dataset
import pickle
from data_processing import TextProcessor, CRNNDataset, collate_fn, transform_eval, transform_train
from utils import post_process, upload_to_hub
from multiprocessing import cpu_count
from model import CRNN
import torch.nn as nn
import torch
import wandb
import random 
from datetime import datetime
import torchvision

hf_token = 'token'
wandb_token = "token"

random.seed(43)
torch.manual_seed(34)


def load_data(batch_size, dataset_name):
    num_workers = cpu_count()
    dataset = load_dataset(dataset_name, num_proc=num_workers, token=hf_token)
    splits = dataset.keys()
    all_text_splits = [dataset[split]['text'] for split in splits]
    all_text = sum(all_text_splits,[])
    chars = [list(ch) for ch in all_text]
    chars_all = [c for char in chars for c in char]
    alphabet = set(chars_all)
    text_processor = TextProcessor(alphabet)
    
    if len(splits)==1:
        train_splits = test_split = 'train'
    else:
        train_splits, test_split = 'train','test'

    #pickle text processor for future use
    with open("text_process.cls","wb") as f:
        pickle.dump(text_processor, f)
    
    #upload to hfhub
    upload_to_hub(file_name='text_process.cls', token=hf_token, commit_message='Uploading text processor')
    

    dset_train = CRNNDataset(height=61, text_processor=text_processor, dataset=dataset[train_splits], transforms=transform_train)
    dset_val = CRNNDataset(height=61, text_processor=text_processor, dataset=dataset[test_split], transforms=transform_eval)

    train_dataloader = DataLoader(
        dset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
        )
    val_dataloader = DataLoader(
        dset_val, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
        )
    
    return train_dataloader, val_dataloader, len(text_processor)


def train_crnn(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (data, labels, target_lengths) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        log_probs = outputs.log_softmax(dim=2)
        log_probs = log_probs.permute(1, 0, 2)
        input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
    return loss.item()


def eval_crnn(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels, target_lengths in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(data)
            log_probs = outputs.log_softmax(dim=2)
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)

            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)


def predict_on_sample(model, eval_dataset:CRNNDataset, device):
    decode = eval_dataset.text_processor.decode
    idx = random.randint(0, len(eval_dataset.dataset))
    inputs = eval_dataset[idx][0]
    to_pil = torchvision.transforms.ToPILImage()
    pil_image = to_pil(inputs)
    inputs = inputs.unsqueeze(0)
    model.eval()
    
    with torch.no_grad():
        predictions = model(inputs.to(device))
    predictions = predictions.detach().cpu()
    pred_ids = torch.argmax(predictions, dim=-1).flatten().tolist()
    text = post_process(decode,pred_ids)
    return pil_image, text


def train(epochs, hidden_size, batch_size, eval_interval, device, dataset_name, lr=1e-3):
    train_dataloader, val_dataloader, num_classes = load_data(dataset_name=dataset_name, batch_size=batch_size)
    model = CRNN(num_channels=1, hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    train_loss = {}
    val_loss = {}
    best_eval_loss = float('inf')

    print("starting model training...")
    for epoch in range(epochs):
        loss = train_crnn(model, train_dataloader, criterion, optimizer, device)
        train_loss[epoch] = loss
        if epoch % eval_interval == 0:
            eval_loss = eval_crnn(model, val_dataloader, criterion, device)
            print(f"Epoch {epoch} | Train Loss: {loss:.3f} | Val Loss: {eval_loss:.3f}")
            infer_image, predicted_text = predict_on_sample(model, eval_dataset=val_dataloader.dataset, device=device)
            log_data = {
                "step": epoch,
                "eval_loss": eval_loss,
                "train_loss": loss,
                "predictions": wandb.Image(
                    infer_image,
                    caption=f"Prediction: {predicted_text}"
                )
            }
            val_loss[epoch] = eval_loss
            wandb.log(log_data)

            if eval_loss < best_eval_loss:
                print("Saveing best model")
                torch.save(model.state_dict(), "crnn.pt")
                upload_to_hub("crnn.pt", token=hf_token, commit_message=f'Upload best model at {str(epoch)}')
                best_eval_loss = eval_loss
            


if __name__ == "__main__":
    #logging related data
    wandb.login(key=wandb_token)
    today = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    wandb_project = "crnn-sinhala"
    wandb_run_name = f"crnn-run-{today}"
    
    dataset_name = "Ransaka/SSOCR-1K"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 100
    batch_size = 1024
    eval_interval = int(epochs/10)
    hidden_size = 256

    #init logger
    config = dict(epochs=epochs, batch_size=batch_size,eval_intervale=eval_interval, hidden_size=hidden_size)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    #start training
    train(
        epochs,
        hidden_size,
        batch_size,
        eval_interval,
        device,
        dataset_name
    )