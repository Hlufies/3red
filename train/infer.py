import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)]) #多模态

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 1024}
    ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in args.mm_emb_id}  # 记录的是不同多模态特征的维度

    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types  #特征跟特征类型
    
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
  
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0  #第0行置0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        pred_list = []
        label_list = []
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            if seq.numel() == 0:  
                print("跳过空批次")
                continue  
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)

            pos_logits, neg_logits, pos_pred, user_indices = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            pos_prediction = torch.nn.Sigmoid()(pos_pred)
            indices = torch.where(next_token_type == 1)
            loss_pos = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss_neg = bce_criterion(neg_logits[indices], neg_labels[indices])
            
            # aux loss
            next_action_type = next_action_type[user_indices[:, 0]].unsqueeze(-1)
            next_token_type = next_token_type[user_indices[:, 0]]

            pos_indices = torch.nonzero((next_token_type == 1))

            pos_pred = pos_pred[pos_indices[:, 0], pos_indices[:, 1]]
            pos_prediction = pos_prediction[pos_indices[:, 0], pos_indices[:, 1]]
            next_action_type = next_action_type[pos_indices[:, 0], pos_indices[:, 1]]

            # 打乱
            perm = torch.randperm(pos_pred.shape[0])
            pos_pred = pos_pred[perm]
            pos_prediction = pos_prediction[perm]
            next_action_type = next_action_type[perm]

            loss_pos_pred = bce_criterion(pos_pred, next_action_type.float()) 
            pred_list.extend(pos_prediction.squeeze(-1).cpu().detach().numpy())
            label_list.extend(next_action_type.squeeze(-1).cpu().detach().numpy())
            loss = loss_pos + loss_neg + loss_pos_pred

            
            
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time(), 'neg_logits': list(neg_logits.shape),'pos_pred[0:20]': list(pos_pred.shape)}
            )

            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            print(pos_pred[0:20].squeeze(-1).cpu().detach().numpy())
            print(pos_prediction[0:20].squeeze(-1).cpu().detach().numpy())
            print(next_action_type[0:20].squeeze(-1).cpu().detach().numpy())

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_pos', loss_pos.item(), global_step)
            writer.add_scalar('Loss/train_neg', loss_neg.item(), global_step)
            writer.add_scalar('Loss/train_pos_pred', loss_pos_pred.item(), global_step)
            writer.add_scalar('Loss/train_pred_mean', np.mean(pred_list), global_step)
            writer.add_scalar('Loss/train_label_mean', np.mean(label_list), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        pred_list = []
        label_list = []
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            if seq.numel() == 0:  
                print("跳过空批次")
                continue
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            next_token_type = next_token_type.to(args.device)
            next_action_type = next_action_type.to(args.device)

            pos_logits, neg_logits, pos_pred, user_indices = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_prediction = torch.nn.Sigmoid()(pos_logits)   
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = torch.where(next_token_type == 1)
            loss_pos = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss_neg = bce_criterion(neg_logits[indices], neg_labels[indices])
            
            next_action_type = next_action_type[user_indices[:, 0]].unsqueeze(-1)
            next_token_type = next_token_type[user_indices[:, 0]]
            pos_indices = torch.nonzero((next_token_type == 1))
            pos_pred = pos_pred[pos_indices[:, 0], pos_indices[:, 1]]
            pos_prediction = pos_prediction[pos_indices[:, 0], pos_indices[:, 1]]
            next_action_type = next_action_type[pos_indices[:, 0], pos_indices[:, 1]]
            loss_pos_pred = bce_criterion(pos_pred, next_action_type.float()) 
            pred_list.extend(pos_prediction.squeeze(-1).cpu().detach().numpy())
            label_list.extend(next_action_type.squeeze(-1).cpu().detach().numpy())
            
            loss = loss_pos + loss_neg + loss_pos_pred
            valid_loss_sum += loss.item()
            
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('Loss/valid_pos', loss_pos.item(), global_step)
        writer.add_scalar('Loss/valid_neg', loss_neg.item(), global_step)
        writer.add_scalar('Loss/valid_pos_pred', loss_pos_pred.item(), global_step)
        writer.add_scalar('Loss/valid_pred_mean', np.mean(pred_list), global_step)
        writer.add_scalar('Loss/valid_label_mean', np.mean(label_list), global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
