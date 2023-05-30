import os
import csv
import time
import torch
import random
import numpy as np
import argparse
import wandb
import pandas as pd
from pathlib import Path
from pyhocon import ConfigFactory, HOCONConverter
from transformers import AdamW, get_linear_schedule_with_warmup

from eval_corefud import conll
from model.data import Dataset, DataLoader
from model.model import Model

from early_stopper import EarlyStopper


class Trainer:

    def __init__(self, conf, gpu, split):
        seed = 126
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.config = ConfigFactory.parse_file('./coref.conf')[conf]
        # configure devices (cpu or up to two sequential gpus)
        use_cuda = gpu and torch.cuda.is_available()
        self.device1 = torch.device('cuda:0' if use_cuda else 'cpu')
        self.device2 = torch.device('cuda:1' if use_cuda else 'cpu')
        self.device2 = self.device2 if split else self.device1
        
        # load dataset with training data
        self.dataset = Dataset(self.config, training=True)
        self.dataloader = DataLoader(self.dataset, shuffle=True)
        
        # load dataset with validation data
        self.val_dataset = Dataset(self.config, validation=True)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False)
        
        # load dataset with test data
        self.test_dataset = Dataset(self.config, testing=True)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=False)

    def train(self, name, amp=False, checkpointing=False):
        # Print infos to console
        print(f"### Start Training ###")
        print(f'running on: {self.device1} {self.device2}')
        print(f'running for: {self.config["epochs"]} epochs')
        print(f'number of batches: {len(self.dataloader)}')
        print(f'saving ckpts to: {name}\n')

        # print full config
        print(HOCONConverter.convert(self.config, 'hocon'))

        # wandb init
        wandb.init(project="coref", entity="rohdas")

        # initialize model and move to gpu if available
        model = Model(self.config, self.device1, self.device2, checkpointing)
        model.bert_model.to(self.device1)
        model.task_model.to(self.device2)
        # model.train()

        # define loss and optimizer
        lr_bert, lr_task = self.config['lr_bert'], self.config['lr_task'],

        # exclude bias and layer-norm from weight decay
        bert_params_wd = []
        bert_params_no_wd = []
        for name_, param in model.bert_model.named_parameters():
            group = bert_params_no_wd if 'bias' in name_ or 'LayerNorm' in name_ else bert_params_wd
            group.append(param)

        # bert fine-tuning
        train_steps = len(self.dataloader) * self.config["epochs"]
        warmup_steps = int(train_steps * 0.1)
        optimizer_bert = AdamW([{'params': bert_params_wd, 'weight_decay': 0.01},
                                {'params': bert_params_no_wd, 'weight_decay': 0.0}], lr=lr_bert, correct_bias=False)
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, warmup_steps, train_steps)

        # task specific optimizer
        optimizer_task = torch.optim.Adam(model.task_model.parameters(), lr=lr_task)
        scheduler_task = get_linear_schedule_with_warmup(optimizer_task, warmup_steps, train_steps)

        # gradient scaler for fp16 training
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # create folder if not already existing
        self.path = Path(f'./data/ckpt/{name}')
        self.path.mkdir(exist_ok=True)
        # load latest checkpoint from path
        epoch = self.load_ckpt(model, optimizer_bert, optimizer_task, scheduler_bert, scheduler_task, scaler)
        
        # prepare csv logging
        csv_path = os.path.join(self.path, self.config["log_file_name"])
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["cur_epoch", "cur_val_f1", "best_epoch", "best_val_f1", "test_f1"]
            writer.writerow(field)
            writer.writerow(["-", "-", "-", "-", "-"])
            
        logging_df = pd.read_csv(csv_path, delimiter=',')
        
        wandb.config = {
            "lr_bert": lr_bert,
            "lr_task": lr_task,
            "epochs": self.config['epochs']
        }

        params_no = sum(param.numel() for param in model.bert_model.parameters() if param.requires_grad)
        params_no += sum(param.numel() for param in model.task_model.parameters() if param.requires_grad)
        
        best_validation_f1 = float('-inf')
        best_epoch = -1
        best_validation_path = ""
        self.patience = self.config['patience']
        early_stopper = EarlyStopper(patience=self.patience)
        
        # run indefinitely until keyboard interrupt
        for e in range(epoch, self.config['epochs']):
            init_epoch_time = time.time()
            train_loss_sum = 0
            
            # train 
            for i, batch in enumerate(self.dataloader):
                model.train()
                optimizer_bert.zero_grad()
                optimizer_task.zero_grad()

                # forward and backward pass
                with torch.cuda.amp.autocast(enabled=amp):
                    scores, labels, _, _, _, _ = model(*batch)
                    loss = self.compute_loss(scores, labels)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer_bert)
                scaler.unscale_(optimizer_task)

                # update weights and lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer_bert)
                scaler.step(optimizer_task)
                scaler.update()
                scheduler_bert.step()
                scheduler_task.step()
                if (i+1) % self.config['log_after'] == 0:
                    print(f'Batch {i+1:04d} of {len(self.dataloader)}', flush=True)
                train_loss_sum += loss
                
            # validate
            with torch.no_grad():
                model.eval()
                coref_preds, subtoken_map = {}, {}
                for j, val_batch in enumerate(self.val_dataloader):
                    # collect data for evaluating batch
                    with torch.cuda.amp.autocast(enabled=amp):
                        _, segm_len, _, _, gold_starts, gold_ends, _, cand_starts, cand_ends, morph_feats, morph_feats_mask, doc_morph_feats = val_batch
                        scores, labels, antes, ment_starts, ment_ends, cand_scores = model(*val_batch)

                    raw_data = self.val_dataset.get_raw_data(j)
                    pred_clusters = self.eval_antecedents(scores, antes, ment_starts, ment_ends, raw_data)
                    coref_preds[raw_data['doc_key']] = pred_clusters
                    subtoken_map[raw_data['doc_key']] = raw_data['token_map']
                
            # evaluate with CorefUD scorer
            corefud_f1 = conll.evaluate_conll(self.config['eval_gold_corefud_path'], self.config['predictions_path'], coref_preds, subtoken_map)
            
            logging_df.loc[0, 'cur_val_f1'] = corefud_f1
            logging_df.loc[0, 'cur_epoch'] = e+1
            
            epoch_time = time.time() - init_epoch_time
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
            print(f'Epoch {e+1:03d} took: {epoch_time}\n', flush=True)
            epoch_loss = train_loss_sum/len(self.dataloader)
            print(f'Loss for Epoch {e:03d}: {epoch_loss}\n', flush=True)
            if e != 0:
                print(f'Best Validation F1 so far: {best_validation_f1} at Epoch {best_epoch:03d}', flush=True)
            print(f'Validation F1 for Epoch {e:03d}: {corefud_f1}\n', flush=True)
            if corefud_f1 > best_validation_f1:    
                # create a checkpoint 
                ckpt_path = self.save_ckpt(e, model, optimizer_bert, optimizer_task, scheduler_bert, scheduler_task, scaler)
                if os.path.exists(best_validation_path):
                    os.remove(best_validation_path)
                best_validation_f1 = corefud_f1
                best_validation_path = ckpt_path
                best_epoch = e
                
                logging_df.loc[0, 'best_val_f1'] = corefud_f1
                logging_df.loc[0, 'best_epoch'] = best_epoch + 1
                
                print("Best validation F1 attained. Saved model checkpoint.\n", flush=True)
            
            logging_df.to_csv(csv_path)
            wandb.log({"loss": epoch_loss})
            
            if early_stopper.early_stop(corefud_f1):
                print(f'No improvement in validation F1 for {self.patience} epochs. Stopping early.', flush=True)
                break
            
        print("Running evaluation on test set.", flush=True)
        
        # load checkpoint with based validation F1
        self.load_ckpt(model, optimizer_bert, optimizer_task, scheduler_bert, scheduler_task, scaler)
        
        # evaluating on test data
        with torch.no_grad():
            model.eval()
            coref_preds, subtoken_map = {}, {}
            for j, test_batch in enumerate(self.test_dataloader):
                # collect data for evaluating batch
                with torch.cuda.amp.autocast(enabled=amp):
                    _, segm_len, _, _, gold_starts, gold_ends, _, cand_starts, cand_ends, morph_feats, morph_feats_mask, doc_morph_feats = test_batch
                    scores, labels, antes, ment_starts, ment_ends, cand_scores = model(*test_batch)

                raw_data = self.test_dataset.get_raw_data(j)
                pred_clusters = self.eval_antecedents(scores, antes, ment_starts, ment_ends, raw_data)
                coref_preds[raw_data['doc_key']] = pred_clusters
                subtoken_map[raw_data['doc_key']] = raw_data['token_map']
                
        # evaluate with CorefUD scorer
        corefud_f1 = conll.evaluate_conll(self.config['test_gold_corefud_path'], self.config['predictions_path'], coref_preds, subtoken_map)
        logging_df.loc[0, 'test_f1'] = corefud_f1
        logging_df.to_csv(csv_path)
        print(f'Test F1: {corefud_f1}\n', flush=True)
        
    def save_ckpt(self, epoch, model, optimizer_bert, optimizer_task, scheduler_bert, scheduler_task, scaler):
        path = self.path.joinpath(f'ckpt_epoch-{epoch:03d}.pt.tar')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_bert': optimizer_bert.state_dict(),
            'optimizer_task': optimizer_task.state_dict(),
            'scheduler_bert': scheduler_bert.state_dict(),
            'scheduler_task': scheduler_task.state_dict(),
            'scaler': scaler.state_dict()
        }, path)

        return path
        

    def load_ckpt(self, model, optimizer_bert, optimizer_task, scheduler_bert, scheduler_task, scaler):
        # check if any checkpoint accessible
        ckpts = list(self.path.glob('ckpt_epoch-*.pt.tar'))
        if not ckpts:
            print(f'\nNo checkpoint found: Start training from scratch\n')
            return 0

        # get latest checkpoint
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_ctime)
        print(f'\nCheckpoint found: Load {latest_ckpt}\n')
        # load latest checkpoint and return next epoch
        latest_ckpt = torch.load(latest_ckpt)
        model.load_state_dict(latest_ckpt['model'])
        optimizer_bert.load_state_dict(latest_ckpt['optimizer_bert'])
        optimizer_task.load_state_dict(latest_ckpt['optimizer_task'])
        scheduler_bert.load_state_dict(latest_ckpt['scheduler_bert'])
        scheduler_task.load_state_dict(latest_ckpt['scheduler_task'])
        scaler.load_state_dict(latest_ckpt['scaler'])
        return latest_ckpt['epoch'] + 1
    
    def eval_antecedents(self, scores, antes, ment_starts, ment_ends, raw_data):
        # tensor to numpy array
        ment_starts = ment_starts.numpy()
        ment_ends = ment_ends.numpy()

        # get best antecedent per mention (as mention index)
        pred_ante_idx = torch.argmax(scores, dim=1) - 1
        pred_antes = [-1 if ante_idx < 0 else antes[ment_idx, ante_idx] for ment_idx, ante_idx in
                      enumerate(pred_ante_idx)]

        # get predicted clusters and mapping of mentions to them
        # antecedents have to be sorted by mention start
        ment_to_pred_cluster = {}
        pred_clusters = []
        for ment_idx, pred_idx in enumerate(pred_antes):
            # ignore dummy antecedent
            if pred_idx < 0:
                continue

            # search for corresponding cluster or create new one
            pred_ante = (ment_starts[pred_idx], ment_ends[pred_idx])
            if pred_ante in ment_to_pred_cluster:
                cluster_idx = ment_to_pred_cluster[pred_ante]
            else:
                cluster_idx = len(pred_clusters)
                pred_clusters.append([pred_ante])
                ment_to_pred_cluster[pred_ante] = cluster_idx

            # add mention to cluster
            ment = (ment_starts[ment_idx], ment_ends[ment_idx])
            pred_clusters[cluster_idx].append(ment)
            ment_to_pred_cluster[ment] = cluster_idx

        # replace mention indices with mention boundaries
        pred_clusters = [tuple(cluster) for cluster in pred_clusters]
        return pred_clusters

    @staticmethod
    def compute_loss(scores, labels):
        # apply mask to get only scores of gold antecedents
        gold_scores = scores + torch.log(labels.float())
        # marginalize gold scores
        gold_scores = torch.logsumexp(gold_scores, [1])
        scores = torch.logsumexp(scores, [1])
        return torch.sum(scores - gold_scores)

    @staticmethod
    def collate(batch):
        return batch[0]


if __name__ == '__main__':
    # parse command line arguments
    folder = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Train e2e coreference resolution model with BERT.')
    parser.add_argument('-c', metavar='CONF', default='bert-base', help='configuration (see coref.conf)')
    parser.add_argument('-f', metavar='FOLDER', default=folder, help='snapshot folder (data/ckpt/<FOLDER>)')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    parser.add_argument('--check', action='store_true', help='use gradient checkpointing')
    parser.add_argument('--split', action='store_true', help='split the model across two GPUs')
    args = parser.parse_args()
    # run training
    Trainer(args.c, not args.cpu, args.split).train(args.f, args.amp, args.check)
