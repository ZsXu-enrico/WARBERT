"""
Training script for WARBERT(R)
- Recommendation-type model with dual-component feature fusion
- Multi-label classification for API recommendation
- Two-phase learning rate schedule
- Early stopping based on validation loss
"""
import os
import time
import random
import pickle

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from tqdm import tqdm

from config import Config
from model import WARBERT_R
from utils.metrics import evaluate_all_metrics, print_metrics


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MashupAPIDataset(Dataset):
    """Dataset for WARBERT(R) training."""

    def __init__(self, data, api_corpus, tokenizer, max_length=256):
        self.data = data
        self.api_corpus = api_corpus
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_apis = len(api_corpus)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mashup_text = item['mashup_desc']

        encoding = self.tokenizer(
            mashup_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Multi-hot label vector
        label = torch.zeros(self.num_apis)
        for api_idx in item['used_apis']:
            if api_idx < self.num_apis:
                label[api_idx] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label,
            'mashup_id': item['mashup_id']
        }


def collate_fn(batch):
    """Custom collate function."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    mashup_ids = [item['mashup_id'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'mashup_ids': mashup_ids
    }


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch."""
    epoch_start_time = time.time()

    model.train()
    total_loss = 0
    num_batches = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    progress_bar = tqdm(train_loader, desc="Training WARBERT(R)")
    loss_fct = nn.BCELoss()

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids, attention_mask, task='api')
        loss = loss_fct(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches

    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    epoch_time = time.time() - epoch_start_time
    return avg_loss, peak_memory_mb, epoch_time


def evaluate(model, data_loader, device):
    """Evaluate model on a data loader. Returns (metrics_dict, avg_loss)."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    loss_fct = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            predictions = model(input_ids, attention_mask, task='api')

            loss = loss_fct(predictions.cpu(), labels)
            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu())
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    results = evaluate_all_metrics(all_predictions, all_labels, k_list=[1, 5, 10])
    avg_loss = total_loss / num_batches
    return results, avg_loss


def main():
    config = Config()
    set_seed(config.SEED)

    print("=" * 60)
    print("WARBERT(R) Training")
    print("=" * 60)

    # Load processed data
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.pkl")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}")
        print("Please run data_preprocessor.py first!")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    api_corpus = data['api_corpus']
    num_apis = data['num_apis']

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test:  {len(test_data)}")
    print(f"  APIs:  {num_apis}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

    # Create datasets and data loaders
    train_dataset = MashupAPIDataset(train_data, api_corpus, tokenizer, config.MAX_LENGTH)
    valid_dataset = MashupAPIDataset(valid_data, api_corpus, tokenizer, config.MAX_LENGTH)
    test_dataset = MashupAPIDataset(test_data, api_corpus, tokenizer, config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_R,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE_R,
                              shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE_R,
                             shuffle=False, collate_fn=collate_fn)

    # Initialize model
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = WARBERT_R(config, num_apis)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (two-phase LR: 1e-3 for first WARMUP_EPOCHS_R epochs, then 1e-5)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE_R,
                      weight_decay=config.WEIGHT_DECAY)

    # Training with early stopping (monitoring val_loss, patience=7)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 7
    total_train_start = time.time()
    best_epoch_time = 0
    model_path = os.path.join(config.OUTPUT_DIR, "warbert_r_best.pt")

    print(f"\nLR schedule: {config.LEARNING_RATE_R} for first "
          f"{config.WARMUP_EPOCHS_R} epochs, 1e-5 for remaining")
    print(f"Early stopping patience: {patience}")
    print("=" * 60)

    for epoch in range(config.NUM_EPOCHS_R):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS_R}")
        print("-" * 60)

        # Switch to second-phase learning rate
        if epoch == config.WARMUP_EPOCHS_R:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            print("Switching learning rate to 1e-5")

        # Train
        train_loss, peak_memory_mb, epoch_time = train_epoch(
            model, train_loader, optimizer, device, config
        )
        print(f"Train Loss: {train_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']}")
        if torch.cuda.is_available():
            print(f"Peak GPU Memory: {peak_memory_mb:.1f} MB")

        # Validate
        val_results, val_loss = evaluate(model, valid_loader, device)
        print(f"Val Loss: {val_loss:.6f}")
        print_metrics(val_results)

        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_epoch_time = time.time() - total_train_start
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"Best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    total_train_time = time.time() - total_train_start

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"  Time to best:  {best_epoch_time:.1f}s ({best_epoch_time / 60:.1f} min)")
    print(f"  Total time:    {total_train_time:.1f}s ({total_train_time / 60:.1f} min)")
    print("=" * 60)

    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results, test_loss = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.6f}")
    print_metrics(test_results)


if __name__ == "__main__":
    main()
