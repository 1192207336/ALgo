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
import torch.nn.functional as F

from dataset import MyDataset
from model import BaselineModel
from torch.optim.lr_scheduler import LambdaLR
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        """
        # TODO: 修改InfoNCE
        """
        # Normalize features
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        # Compute similarities
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        neg_sim = (anchor * negative).sum(dim=-1) / self.temperature
        # Concatenate similarities
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch_size, 2]
        # Labels: 0 for positive
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        # Compute InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        return loss

def compute_contrastive_loss(user_feats, pos_embs, pos_ids, neg_embs, temperature, l2_emb, model):
    """
    user_feats: [B, D]
    pos_embs:   [B, D]
    pos_ids:    [B] (item ids for masking)
    neg_embs:   [B * num_neg, D]  (flattened neg embeddings)
    """

    # 归一化
    user_feats = F.normalize(user_feats, dim=-1)
    pos_embs = F.normalize(pos_embs, dim=-1)
    if neg_embs is not None and neg_embs.numel() > 0:
        neg_embs = F.normalize(neg_embs, dim=-1)

    # 拼接 candidates
    if neg_embs is not None and neg_embs.numel() > 0:
        candidates = torch.cat([pos_embs, neg_embs], dim=0)  # [B + B*num_neg, D]
    else:
        candidates = pos_embs  # [B, D]

    # logits: [B, num_candidates]
    logits = torch.matmul(user_feats, candidates.T) / temperature

    # Mask 重复 item（只对前B列 in-batch 部分做，因为neg_embs来自dataset通常不和pos重复）
    B = pos_embs.size(0)
    eq_mask = pos_ids.unsqueeze(1) == pos_ids.unsqueeze(0)  # [B, B]
    eq_mask.fill_diagonal_(False)  # 保留自己的正样本
    logits[:, :B].masked_fill_(eq_mask.to(logits.device), -1e9)

    # labels: 正样本在 candidates 前B列的对应位置
    labels = torch.arange(B, device=logits.device)

    loss = F.cross_entropy(logits, labels)

    # L2 正则（平方形式）
    for param in model.item_emb.parameters():
        loss = loss + l2_emb * torch.norm(param) ** 2

    return loss



def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    # TODO: 修改数据集，使其能采样不同数量的负样本
    parser.add_argument('--num_negatives', default=4, type=int)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args

def linear_warmup(step, warmup_steps, max_lr):
    """Linear warmup scheduler."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    criterion = InfoNCE(temperature=args.temperature)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, args.warmup_steps, args.lr))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in loop:
            optimizer.zero_grad()
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            # pos_logits, neg_logits = model(
            #     seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            # )
            # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
            #     neg_logits.shape, device=args.device
            # )
            # indices = np.where(next_token_type == 1)

            # 取最后一个时间步的 item id 作为 pos_ids（避免泄露）
            pos_ids = pos[:, -1]  # shape [B]
            # forward
            user_feats = model.predict(seq, seq_feat, token_type)   # [B, D]
            pos_embs = model.feat2emb(pos, pos_feat, include_user=False)[:, -1, :]  # [B, D]
            # 处理 neg_embs
            if neg.dim() == 3:  # [B, num_neg, seq_len]
                B, K, _ = neg.shape
                neg_embs = model.feat2emb(neg.view(B * K, -1), neg_feat.view(B * K, -1, neg_feat.size(-1)), include_user=False)[:, -1, :]
            else:  # [B, seq_len]
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)[:, -1, :]

            loss = compute_contrastive_loss(user_feats, pos_embs, pos_ids, neg_embs, args.temperature, args.l2_emb, model)

            # loss = criterion(pos_logits[indices], pos_labels[indices])
            # loss += criterion(neg_logits[indices], neg_labels[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                # loss += args.l2_emb * torch.norm(param) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()
            # scheduler.step(epoch)

        model.eval()
        valid_loss_sum = 0
        for step, batch in enumerate(valid_loader):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            # pos_logits, neg_logits = model(
            #     seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            # )
            # pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
            #     neg_logits.shape, device=args.device
            # )
            # indices = np.where(next_token_type == 1)

            # loss = criterion(pos_logits[indices], pos_labels[indices])
            # loss += criterion(neg_logits[indices], neg_labels[indices])

            pos_ids = pos[:, -1]
            user_feats = model.predict(seq, seq_feat, token_type)
            pos_embs = model.feat2emb(pos, pos_feat, include_user=False)[:, -1, :]

            if neg.dim() == 3:
                B, K, _ = neg.shape
                neg_embs = model.feat2emb(neg.view(B * K, -1), neg_feat.view(B * K, -1, neg_feat.size(-1)),
                                          include_user=False)[:, -1, :]
            else:
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)[:, -1, :]

            loss = compute_contrastive_loss(user_feats, pos_embs, pos_ids, neg_embs, args.temperature, args.l2_emb, model)
            valid_loss_sum += loss.item()

        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        print(f"Validation loss: {valid_loss_sum}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
