import argparse
from pathlib import Path
from functools import partial
import json
import math
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import transformers
transformers.logging.set_verbosity_error()

from pytorch_lightning.utilities.seed import seed_everything

MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL)

class PointDataset(Dataset):

    def __init__(self, file_dir):
        self.examples = []
        with open(file_dir, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))

        print('loaded {} examples from {}'.format(len(self.examples), file_dir))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return [x['response'], x['concern']['text'], x['opinion']['text']], x['evaluation']['mean_score']

class PointCSVDataset(Dataset):
        def __init__(self, file_dir):
            self.examples = []
            df = pd.read_csv(file_dir)
            for i, row in df.iterrows():
                self.examples.append(row['content'])

            print('loaded {} examples from {}'.format(len(self.examples), file_dir))

        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            return [self.examples[idx]], 0

class PairDataset(Dataset):

    def __init__(self, file_dir):
        self.examples = []
        with open(file_dir, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))

        # for i in range(len(self.examples)-1, -1, -1):
        #     if self.examples[i]['majority_vote'] == '?':
        #         del self.examples[i]

        print('loaded {} pairs from {}'.format(len(self.examples), file_dir))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        dataA = [x['responseA']['response'], x['responseA']['concern']['text'], x['responseA']['opinion']['text']]
        dataB = [x['responseB']['response'], x['responseB']['concern']['text'], x['responseB']['opinion']['text']]
        if x['majority_vote'] == 'A':
            return dataA, dataB
        else:
            return dataB, dataA

def point_collate(samples, max_len=500):

    response = []
    labels = []

    for example, label in samples:
        response.append(' [SEP] '.join(example))
        labels.append(label)
    
    response = tokenizer(response, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    labels = torch.tensor(labels)

    return {
        'response': response,
        'labels': labels,
    }

def pair_collate(samples, max_len=500):

    response = []
    labels = []

    for example_A, example_B in samples:
        response.append(' [SEP] '.join(example_A))
        response.append(' [SEP] '.join(example_B))
        labels.append(0)
        labels.append(0)
    
    response = tokenizer(response, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    labels = torch.tensor(labels)

    return {
        'response': response,
        'labels': labels,
    }


class Finetuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.bert = BertModel.from_pretrained(MODEL)
        self.regression = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, batch, predict=False):
        batch_size = batch['response']['input_ids'].shape[0]

        # get embeddings from BERT
        outputs = self.bert(output_attentions=False,
                        output_hidden_states=False,
                        **batch['response'],
                        )
        instruction_states = outputs.last_hidden_state[:, 0]
        instruction_states = self.dropout(instruction_states)
        scores = self.regression(instruction_states).squeeze()

        if predict:
            return { 'scores': scores }

        # compute MSE loss, for training
        # loss = F.mse_loss(scores, batch['labels'].float())

        # compute margin loss, for continuous fine-tuning
        margin = 0.5
        loss = F.margin_ranking_loss(scores[::2], scores[1::2], torch.ones(batch_size//2).to(scores.device), margin=margin)

        # check nan
        if torch.isnan(loss).any():
            loss = torch.tensor(0.0).to(loss.device)

        return {
            'scores': scores,
            'loss': loss
        }


    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        return {
            'val_loss': output['loss'],
            'scores': output['scores'],
            'labels': batch['labels'],
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        scores = torch.cat([x['scores'] for x in outputs])

        scores = scores.cpu().tolist()
        count = 0
        for i in range(0, len(scores), 2):
            count += scores[i] > scores[i+1]
        acc = count / (len(scores) / 2)

        if self.global_rank == 0:
            tensorboard = self.logger.experiment
        
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': loss}
    
    def predict_epoch_end(self, outputs):
        scores = torch.cat([x['scores'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        scores = scores.cpu().tolist()
        results = {'scores': scores }
        with open('predictions.json', 'w') as f:
            json.dump(results, f)
        
        return results

    # for absolute ranking
    # def train_dataloader(self):
    #     return DataLoader(PointDataset(self.hparams.train_dataset),
    #                       self.hparams.batch_size,
    #                       shuffle=True,
    #                       collate_fn=point_collate,
    #                       num_workers=0)


    # for relative ranking
    def train_dataloader(self):
        return DataLoader(PairDataset(self.hparams.train_dataset),
                          self.hparams.batch_size,
                          shuffle=True,
                          collate_fn=pair_collate,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(PairDataset(self.hparams.val_dataset),
                          self.hparams.batch_size,
                          shuffle=False,
                          collate_fn=pair_collate,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(PairDataset(self.hparams.test_dataset),
                          self.hparams.batch_size,
                          shuffle=False,
                          collate_fn=pair_collate,
                          num_workers=0)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        pgs = [{'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01},
               {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}]

        bert_optimizer = AdamW(pgs, lr=self.hparams.lr)
        bert_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(bert_optimizer, self.hparams.max_steps // 10, self.hparams.max_steps),
            'interval': 'step',
            'monitor': None
        }
        return [bert_optimizer], [bert_scheduler]


GPT_DATA_BASE_PATH = 'data/auto/v1/'
FILTER_GPT_DATA_BASE_PATH = 'data/auto/v1-filtered/'
HUMAN_DATA_BASE_PATH = 'data/annotated/'

hparams = {
    'batch_size': None,
    'max_epochs': None,
    # 'train_dataset': GPT_DATA_BASE_PATH + 'train_set-absolute.jsonl', # for training (regression)
    # 'train_dataset': GPT_DATA_BASE_PATH + 'train_set-relative.jsonl', # for training (ranking)
    # 'train_dataset': FILTER_GPT_DATA_BASE_PATH + 'train_set-absolute.jsonl', # for filtered training
    'train_dataset': HUMAN_DATA_BASE_PATH + 'train_set-relative.jsonl', # for continue fine-tuning
    'val_dataset': HUMAN_DATA_BASE_PATH + 'dev_set-relative.jsonl',
    'test_dataset': HUMAN_DATA_BASE_PATH + 'test_set-relative.jsonl'
}

if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", default='default', nargs='?', help="Name of the experiment")
    parser.add_argument("--mode", default='train', help="train or eval")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
    parser.add_argument("--max_epochs", default=5, type=int, help="Number of epochs")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="Number of batches to accumulate gradients for")
    args = parser.parse_args()

    hparams['batch_size'] = args.batch_size
    hparams['lr'] = args.lr
    hparams['dropout'] = args.dropout
    hparams['max_epochs'] = args.max_epochs

    if args.mode == 'train':
        if args.ckpt:
            model = Finetuner.load_from_checkpoint(args.ckpt, **hparams) # for continue fine-tuning
        else:
            model = Finetuner(hparams)

        effective_batch_size = hparams['batch_size'] * args.accumulate_grad_batches * args.gpus
        model.hparams['max_steps'] = math.ceil(len(model.train_dataloader().dataset) / effective_batch_size) * hparams['max_epochs']
        logger = TensorBoardLogger('local/tensorboard', args.exp_name)
        trainer = pl.Trainer(devices=args.gpus, accelerator="gpu", auto_select_gpus=True,
                            max_epochs=hparams['max_epochs'], max_steps=model.hparams['max_steps'],
                            accumulate_grad_batches=args.accumulate_grad_batches,
                            gradient_clip_val=1.0, logger=logger,
                            callbacks=[
                                LearningRateMonitor(),
                                ModelCheckpoint(dirpath=f'local/{args.exp_name}-ckpt', filename='{epoch}-{val_acc:.2f}', monitor='val_acc', mode='max', save_top_k=1, save_last=True, save_weights_only=True),
                            ],
                            log_every_n_steps=10,
                            precision=16,)
        trainer.fit(model)
    elif args.mode == 'eval':
        model = Finetuner.load_from_checkpoint(args.ckpt, **hparams)
        trainer = pl.Trainer(gpus=1, accelerator="cuda", precision=16)
        predictions = trainer.predict(model, dataloaders=model.test_dataloader())
        predictions = model.predict_epoch_end(predictions)
        print(predictions)
