import argparse
import os
import time
import torch
import numpy as np
import random
from utils import str2bool, AverageMeter, accuracy
from dataloader import  get_dataloader
from model import get_model

from arguments import get_args


def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    args = get_args()

    _, test_dataloader= get_dataloader(args)
    model = get_model(args).cuda()

    
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    test_acc_recoder = AverageMeter()
    test_loss_recoder = AverageMeter()
    
    # if test_acc_recoder.avg > best_test_acc:
    # best_test_acc = test_acc_recoder.avg
    # state = {
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'val_accuracy': best_test_acc,
    # }
    # torch.save(state, args.ckpt_save_path +
    #             f'/valBest_{best_test_acc:.3f}_ckpt.pth.tar')
        
    ckpt = torch.load(args.ckpt_load_path)

    test_loss_recoder.reset()
    test_acc_recoder.reset()
    '''
        Test!
    '''
    model.eval()
    with torch.no_grad():
        for input, action in test_dataloader:
            input,action = input.cuda(),action.cuda()
            prediction = model(input)
            
            val_loss = loss_function(prediction, action)
            val_acc = accuracy(prediction, action)

            test_loss_recoder.update(val_loss.item(), n=action.size(0))
            test_acc_recoder.update(val_acc.item(), n=action.size(0))

    print(f'test acc:{test_acc_recoder.avg:.4f} test loss:{test_loss_recoder.avg:.4f}')


if __name__ == '__main__':
    main()