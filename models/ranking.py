import argparse
from pathlib import Path
from functools import partial
import json
import math
import itertools

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import transformers
transformers.logging.set_verbosity_error()

seed_everything(42)

MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL)

class RankingDataset(Dataset):

    def __init__(self, file_dir):
        self.examples = []
        with open(file_dir, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))

        for i in range(len(self.examples)-1, -1, -1):
            if self.examples[i]['majority_vote'] == '?':
                del self.examples[i]

        print('loaded {} examples from {}'.format(len(self.examples), file_dir))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return (x['responseA']['response'], x['responseB']['response']) , int(x['majority_vote'] == 'A')



def ranking_collate(samples, max_len=500):

    response_A = []
    response_B = []
    response_AB = []
    response_BA = []
    labels = []

    for example, label in samples:
        response_A.append(example[0])
        response_B.append(example[1])
        response_AB.append(example[0] + ' [SEP] ' + example[1])
        response_BA.append(example[1] + ' [SEP] ' + example[0])
        labels.append(label)
    
    # tokenize and print length
    # inputs = tokenizer(response_AB)
    # print([len(x) for x in inputs['input_ids']])
    # print('max length: {}'.format(max([len(x) for x in inputs['input_ids']])))

    response_A = tokenizer(response_A, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    response_B = tokenizer(response_B, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    response_AB = tokenizer(response_AB, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    response_BA = tokenizer(response_BA, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    return {
        'response_A': response_A,
        'response_B': response_B,
        'response_AB': response_AB,
        'response_BA': response_BA,
        'labels': torch.tensor(labels)
    }



class Finetuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.bert = BertModel.from_pretrained(MODEL)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, batch):
        batch_size = batch['response_A']['input_ids'].shape[0]

        # get embeddings from BERT
        outputs_AB = self.bert(output_attentions=False,
                        output_hidden_states=False,
                        **batch['response_AB'],
                        )
        outputs_BA = self.bert(output_attentions=False,
                        output_hidden_states=False,
                        **batch['response_BA'],
                        )
        scores = self.classifier(self.dropout(outputs_AB.last_hidden_state[:, 0])) - self.classifier(self.dropout(outputs_BA.last_hidden_state[:, 0]))

        # compute loss
        loss = F.cross_entropy(scores, batch['labels'])
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

        if self.global_rank == 0:
            tensorboard = self.logger.experiment
        
        acc = (scores.argmax(dim=1) == torch.cat([x['labels'] for x in outputs])).float().mean()
        # print('predictions')
        # print(scores.argmax(dim=1))
        # print('labels')
        # print(torch.cat([x['labels'] for x in outputs]))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': loss}
    
    def predict_epoch_end(self, outputs):
        scores = torch.cat([x['scores'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])

        predictions = scores.argmax(dim=1)
        acc = (predictions == labels).float().mean().item()

        results = {
                'acc': acc,
                'scores': scores.cpu().tolist(),
                    }
        print(results)
        with open('predictions.json', 'w') as f:
            json.dump(results, f)

    def train_dataloader(self):
        return DataLoader(RankingDataset(self.hparams.train_dataset),
                          self.hparams.batch_size,
                          shuffle=True,
                          collate_fn=ranking_collate,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(RankingDataset(self.hparams.val_dataset),
                          self.hparams.batch_size,
                          shuffle=False,
                          collate_fn=ranking_collate,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(RankingDataset(self.hparams.test_dataset),
                          self.hparams.batch_size,
                          shuffle=False,
                          collate_fn=ranking_collate,
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
HUMAN_DATA_BASE_PATH = 'data/annotated/'

hparams = {
    'batch_size': None,
    'max_epochs': None,
    'train_dataset': GPT_DATA_BASE_PATH + 'train_set-relative.jsonl',
    'val_dataset': HUMAN_DATA_BASE_PATH + 'dev_set-relative.jsonl',
    'test_dataset': HUMAN_DATA_BASE_PATH + 'test_set-relative.jsonl',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", default='default', nargs='?', help="Name of the experiment")
    parser.add_argument("--mode", default='train', help="train or eval")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint")
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--max_epochs", default=5, type=int, help="Number of epochs")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="Number of batches to accumulate gradients for")
    args = parser.parse_args()

    hparams['batch_size'] = args.batch_size
    hparams['lr'] = args.lr
    hparams['dropout'] = args.dropout
    hparams['max_epochs'] = args.max_epochs

    if args.mode == 'train':
        if args.ckpt:
            model = Finetuner.load_from_checkpoint(args.ckpt, hparams)
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
                                ModelCheckpoint(dirpath=f'local/{args.exp_name}-ckpt', filename='{epoch}-{val_acc:.2f}', monitor='val_acc', mode='max', save_top_k=1, save_last=True),
                            ],
                            log_every_n_steps=10,
                            precision=16,)
        trainer.fit(model)
    elif args.mode == 'eval':
        model = Finetuner.load_from_checkpoint(args.ckpt, hparams)
        trainer = pl.Trainer(gpus=1, accelerator="cuda", precision=16)
        predictions = trainer.predict(model, dataloaders=model.test_dataloader())
        print(len(predictions))
        model.predict_epoch_end(predictions)
