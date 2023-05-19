import argparse
import os
import time
import torch
import numpy as np
import random
from utils import  AverageMeter, accuracy
from dataloader import  get_dataloader
from model import get_model
from arguments import get_args
import torch.nn as nn
import torch.nn.functional as F

def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    args = get_args()

    train_dataloader, val_dataloader= get_dataloader(args)
    model = get_model().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epoch,
    )
    
    loss_function = Multi_Class_Focal_Loss(
        masks=torch.ones((5,1)),
        
        ).cuda()
    # loss_function = FocalLoss().cuda()
    
    # loss_function = torch.nn.CrossEntropyLoss().cuda()

    best_val_acc = 0.0

    train_acc_recoder = AverageMeter()
    train_loss_recoder = AverageMeter()
    val_acc_recoder = AverageMeter()
    val_loss_recoder = AverageMeter()
        
    f = open(os.path.join(args.ckpt_save_path, 'train record.csv'), 'w')
    
    '''
        Fitting!
    '''
    for epoch in range(1, args.epoch+1):
        train_loss_recoder.reset()
        train_acc_recoder.reset()
        val_loss_recoder.reset()
        val_acc_recoder.reset()

        '''
            Training!
        '''
        model.train()
        for input, action in train_dataloader:
            input, action = input.cuda(), action.cuda()
            
            prediction = model(input)
            train_loss = loss_function(prediction, action)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            train_acc = accuracy(prediction, action)

            train_loss_recoder.update(train_loss.item(), n=action.size(0))
            train_acc_recoder.update(train_acc.item(), n=action.size(0))

        '''
            Validation!
        '''
        model.eval()
        with torch.no_grad():
            for input, action in val_dataloader:
                input,action = input.cuda(),action.cuda()

                prediction = model(input)
                val_loss = loss_function(prediction, action)
                val_acc = accuracy(prediction, action)

                val_loss_recoder.update(val_loss.item(), n=action.size(0))
                val_acc_recoder.update(val_acc.item(), n=action.size(0))
        '''
            Logging!
        '''
        
        msg = f'Epoch,{epoch},train loss,{train_loss_recoder.avg:.4f},train acc,{train_acc_recoder.avg:.2f},val loss,{val_loss_recoder.avg:.4f},val acc,{val_acc_recoder.avg:.2f},best val acc,{best_val_acc:.3f}'
        
        print(msg), print(msg, file=f)
        
        if val_acc_recoder.avg > best_val_acc:
            best_val_acc = val_acc_recoder.avg
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_accuracy': best_val_acc,
            }
            torch.save(state, args.ckpt_save_path +
                       f'/valBest_{best_val_acc:.3f}_ckpt.pth.tar')

            
        scheduler.step()
        
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_accuracy': best_val_acc,
    }
    torch.save(state, args.ckpt_save_path +
                f'/Final_{val_acc_recoder.avg:.3f}_ckpt.pth.tar')
    f.close()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Multi_Class_Focal_Loss(nn.Module):
    def __init__(self, masks:torch.Tensor, alpha:torch.Tensor|None, gamma: float, reduction:str):
        self.masks = masks
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        
    def forward(self, inputs, targets):
        if self.masks.sum().detach().cpu().item() != 0:
            n_classes = inputs.shape[1]
            logit = F.softmax(inputs, dim=1)  # TENSOR (B x N, C)
            if alpha is None:
                alpha = torch.ones((n_classes), requires_grad=False)

            if alpha.device != inputs.device:
                alpha = alpha.to(inputs.device)

            epsilon = 1e-10
            pt = torch.sum((targets * logit), dim=1, keepdim=True) + epsilon  # TENSOR (B x N, 1)
            log_pt = pt.log()  # TENSOR (B x N, 1)

            targets_idx = torch.argmax(targets, dim=1, keepdim=True).long()  # TENSOR (B x N, 1)
            alpha = alpha[targets_idx]  # TENSOR ( B x N, 1)

            focal_loss = -1 * alpha * (torch.pow((1 - pt), self.gamma) * log_pt)  # TENSOR (B x N, 1)
            masked_focal_loss = focal_loss * self.masks  # TENSOR (B x N, 1)

            if self.reduction == "mean":
                loss = masked_focal_loss.sum() / self.masks.sum()
            elif self.reduction == "sum":
                loss = masked_focal_loss.sum()
            else:
                loss = masked_focal_loss

            return loss
        else:
            return torch.Tensor([0.0]).float().to(inputs.device)
    
    
def multiclass_focal_loss(
    inputs: torch.Tensor,  # TENSOR (B x N, C)
    targets: torch.Tensor,  # TENSOR (B x N, C)
    masks: torch.Tensor,  # TENSOR (B x N, 1)
    alpha=None,  # TENSOR (C, 1)
    gamma: float = 2,
    reduction: str = "none",
):
    if masks.sum().detach().cpu().item() != 0:
        n_classes = inputs.shape[1]
        logit = F.softmax(inputs, dim=1)  # TENSOR (B x N, C)
        if alpha is None:
            alpha = torch.ones((n_classes), requires_grad=False)

        if alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)

        epsilon = 1e-10
        pt = torch.sum((targets * logit), dim=1, keepdim=True) + epsilon  # TENSOR (B x N, 1)
        log_pt = pt.log()  # TENSOR (B x N, 1)

        targets_idx = torch.argmax(targets, dim=1, keepdim=True).long()  # TENSOR (B x N, 1)
        alpha = alpha[targets_idx]  # TENSOR ( B x N, 1)

        focal_loss = -1 * alpha * (torch.pow((1 - pt), gamma) * log_pt)  # TENSOR (B x N, 1)
        masked_focal_loss = focal_loss * masks  # TENSOR (B x N, 1)

        if reduction == "mean":
            loss = masked_focal_loss.sum() / masks.sum()
        elif reduction == "sum":
            loss = masked_focal_loss.sum()
        else:
            loss = masked_focal_loss

        return loss
    else:
        return torch.Tensor([0.0]).float().to(inputs.device)
    
if __name__ == '__main__':
    main()