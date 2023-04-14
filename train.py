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

def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    args = get_args()

    train_dataloader, val_dataloader= get_dataloader(args)
    model = get_model(args).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    best_test_acc = 0.0

    train_acc_recoder = AverageMeter()
    train_loss_recoder = AverageMeter()
    test_acc_recoder = AverageMeter()
    test_loss_recoder = AverageMeter()
        
    f = open(os.path.join(args.ckpt_save_path, 'train record.csv'), 'w')
    
    '''
        Fitting!
    '''
    for epoch in range(1, args.epoch+1):
        train_loss_recoder.reset()
        train_acc_recoder.reset()
        test_loss_recoder.reset()
        test_acc_recoder.reset()

        '''
            Training!
        '''
        model.train()
        for input, action in train_dataloader:
            input,action = input.cuda(),action.cuda()
            
            prediction = model(input)
            train_loss = loss_function(prediction, action)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            train_acc = accuracy(prediction, action)

            train_loss_recoder.update(train_loss.item(), n=action.size(0))
            train_acc_recoder.update(train_acc.item(), n=action.size(0))

        '''
            Test!
        '''
        model.eval()
        with torch.no_grad():
            for input, action in val_dataloader:
                input,action = input.cuda(),action.cuda()

                
                # with autocast():
                prediction = model(input)
                val_loss = loss_function(prediction, action)
                
                val_acc = accuracy(prediction, action)

                test_loss_recoder.update(val_loss.item(), n=action.size(0))
                test_acc_recoder.update(val_acc.item(), n=action.size(0))

        '''
            Logging!
        '''
        msg = f'Epoch,{epoch},train loss,{train_loss_recoder.avg:.4f},train acc,{train_acc_recoder.avg:.2f},test loss,{test_loss_recoder.avg:.4f},test acc,{test_acc_recoder.avg:.2f},best test acc,{best_test_acc:.3f}'
        
        print(msg), print(msg, file=f)
        
        if test_acc_recoder.avg > best_test_acc:
            best_test_acc = test_acc_recoder.avg
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_accuracy': best_test_acc,
            }
            torch.save(state, args.ckpt_save_path +
                       f'/valBest_{best_test_acc:.3f}_ckpt.pth.tar')

            
        scheduler.step()
        
    f.close()

if __name__ == '__main__':
    main()