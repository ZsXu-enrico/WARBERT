"""
Training script for WARBERT(M) 
Attention Comparison
Match-type pairwise mashup-API matching.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
import numpy as np
import pickle
from tqdm import tqdm
import random

from config import Config
from model import WARBERT_M
from data_preprocessor import create_negative_samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MashupAPIPairDataset(Dataset):
    """
    Pairwise dataset for WARBERT(M).
    Each sample: (mashup, API, label), with negative sampling.
    """

    def __init__(self, mashup_data, api_corpus, tokenizer, max_length=256,
                 num_negatives=5, hard_negative_scores=None, hard_ratio=0.3):
        self.mashup_data = mashup_data
        self.api_corpus = api_corpus
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.hard_negative_scores = hard_negative_scores
        self.hard_ratio = hard_ratio
        self.mashup_id_to_idx = {m['mashup_id']: i for i, m in enumerate(mashup_data)}
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for train_idx, mashup_entry in enumerate(tqdm(self.mashup_data, desc="Creating samples")):
            mashup_id = mashup_entry['mashup_id']
            positive_apis = mashup_entry['used_apis']
            mashup_hard_scores = self.hard_negative_scores[train_idx] if self.hard_negative_scores is not None else None

            for api_idx in positive_apis:
                samples.append((mashup_id, api_idx, 1))

            negatives = create_negative_samples(
                mashup_entry, self.api_corpus,
                num_negatives=self.num_negatives * len(positive_apis),
                hard_negative_scores=mashup_hard_scores,
                hard_ratio=self.hard_ratio
            )
            for api_idx in negatives:
                samples.append((mashup_id, api_idx, 0))

        pos = sum(1 for _, _, l in samples if l == 1)
        neg = sum(1 for _, _, l in samples if l == 0)
        print(f"Samples: {len(samples)} total ({pos} pos, {neg} neg)")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mashup_id, api_idx, label = self.samples[idx]
        mashup = self.mashup_data[self.mashup_id_to_idx[mashup_id]]
        api = self.api_corpus[api_idx]

        mashup_text = mashup['mashup_desc']
        api_text = api['api_desc']

        encoding = self.tokenizer(
            mashup_text, api_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
    }


def get_hard_negative_scores(warbert_r_path, mashup_data, api_corpus, tokenizer, device, config):
    """Use WARBERT(R) predictions as hard negative scores for training."""
    from model import WARBERT_R
    from train_warbert_r import MashupAPIDataset, collate_fn as collate_fn_r

    print("Generating hard negative scores from WARBERT(R)...")
    checkpoint = torch.load(warbert_r_path, map_location='cpu', weights_only=False)
    model = WARBERT_R(config, len(api_corpus))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    temp_dataset = MashupAPIDataset(mashup_data, api_corpus, tokenizer, config.MAX_LENGTH)
    temp_loader = DataLoader(temp_dataset, batch_size=config.BATCH_SIZE_R,
                             shuffle=False, collate_fn=collate_fn_r)

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(temp_loader, desc="WARBERT(R) scoring"):
            preds = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), task='api')
            all_predictions.append(preds.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"Hard negative scores shape: {all_predictions.shape}")
    return all_predictions.numpy()


def train_epoch(model, train_loader, optimizer, device, config):
    import time
    model.train()
    total_loss = 0
    loss_fct = nn.BCELoss()
    start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fct(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0
    return total_loss / len(train_loader), peak_memory_mb, time.time() - start


def evaluate_m(model, eval_data, api_corpus, tokenizer, device, config):
    """Score all APIs for each mashup and compute ranking metrics."""
    model.eval()
    num_apis = len(api_corpus)
    api_texts = [api['api_desc'] for api in api_corpus]

    all_scores, all_labels = [], []

    for mashup_entry in tqdm(eval_data, desc="Evaluating"):
        mashup_text = mashup_entry['mashup_desc']
        label = torch.zeros(num_apis)
        for api_idx in mashup_entry['used_apis']:
            if api_idx < num_apis:
                label[api_idx] = 1.0
        all_labels.append(label)

        mashup_scores = torch.zeros(num_apis)
        for start in range(0, num_apis, config.BATCH_SIZE_M):
            end = min(start + config.BATCH_SIZE_M, num_apis)
            encodings = tokenizer(
                [mashup_text] * (end - start), api_texts[start:end],
                max_length=config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                scores = model(
                    encodings['input_ids'].to(device),
                    encodings['attention_mask'].to(device),
                    encodings['token_type_ids'].to(device)
                )
                mashup_scores[start:end] = scores.cpu()
        all_scores.append(mashup_scores)

    all_scores = torch.stack(all_scores)
    all_labels = torch.stack(all_labels)

    from utils.metrics import evaluate_all_metrics
    return evaluate_all_metrics(all_scores, all_labels, k_list=config.TOP_K)


def main():
    config = Config()
    set_seed(config.SEED)

    print("WARBERT(M) Training")
    print(f"Negatives per positive: {config.NUM_NEGATIVE_SAMPLES}, hard ratio: {config.HARD_NEGATIVE_RATIO}")

    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.pkl")
    if not os.path.exists(data_path):
        print(f"Processed data not found at {data_path}. Run data_preprocessor.py first.")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_data = data['train']
    valid_data = data['valid']
    test_data  = data['test']
    api_corpus = data['api_corpus']
    num_apis   = data['num_apis']

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}, APIs: {num_apis}")

    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    warbert_r_path = os.path.join(config.OUTPUT_DIR, "warbert_r_best.pt")
    hard_negative_scores = None
    if os.path.exists(warbert_r_path):
        hard_negative_scores = get_hard_negative_scores(
            warbert_r_path, train_data, api_corpus, tokenizer, device, config
        )
    else:
        print("WARBERT(R) checkpoint not found. Using random negative sampling only.")

    print("Creating training dataset...")
    train_dataset = MashupAPIPairDataset(
        train_data, api_corpus, tokenizer, config.MAX_LENGTH,
        num_negatives=config.NUM_NEGATIVE_SAMPLES,
        hard_negative_scores=hard_negative_scores,
        hard_ratio=config.HARD_NEGATIVE_RATIO
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_M,
                              shuffle=True, collate_fn=collate_fn)

    model = WARBERT_M(config)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE_M, weight_decay=config.WEIGHT_DECAY)

    best_val_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(config.OUTPUT_DIR, "warbert_m_best.pt")

    import time
    total_start = time.time()

    for epoch in range(config.NUM_EPOCHS_M):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS_M}")

        if epoch == config.WARMUP_EPOCHS_M:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            print("  Switching learning rate to 1e-5")

        train_loss, peak_mem, epoch_time = train_epoch(model, train_loader, optimizer, device, config)
        print(f"  Train loss: {train_loss:.4f} | Time: {epoch_time:.1f}s" +
              (f" | Peak GPU: {peak_mem:.0f}MB" if torch.cuda.is_available() else ""))

        val_results = evaluate_m(model, valid_data, api_corpus, tokenizer, device, config)
        val_metric = val_results['NDCG@5']
        print(f"  Val NDCG@5: {val_metric:.4f} | NDCG@10: {val_results['NDCG@10']:.4f} | Rec@5: {val_results['Recall@5']:.4f}")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, best_model_path)
            print(f"  -> Best model saved (Val NDCG@5: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    total_time = time.time() - total_start
    print(f"\nTraining complete. Best epoch: {best_epoch}, Val NDCG@5: {best_val_metric:.4f}")
    print(f"Total time: {total_time/60:.1f} min")

    # Load best model for test evaluation
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    from utils.metrics import print_metrics
    print("\nWARBERT(M) Test Set Evaluation")
    test_results = evaluate_m(model, test_data, api_corpus, tokenizer, device, config)
    print_metrics(test_results)

    results_path = os.path.join(config.OUTPUT_DIR, "warbert_m_test_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()
