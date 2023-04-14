import argparse
import os
import time
import torch
import numpy as np
import random
from utils import str2bool, AverageMeter, accuracy
from dataloader import  get_dataloader
from model import get_model


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='name')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default='GTA', choices=['MCD', 'GTA'])
    ''' dataloader '''
    parser.add_argument('--data_root', type=str, default='./data/processed-data')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--single_individual', type=int ,default=1)
    
    ''' model and network '''
    parser.add_argument('--lr', type=float, default=0.01)    
    parser.add_argument('--weight_decay', type=float, default=1e-5)    
    parser.add_argument('--network', type=str, default='baseline', choices=['baseline', 'F+C+D-f', 'F+C+G+D-f', 'F+C+G+D-g'])
    
    
    parser.add_argument('--ckpt_save_path', type=str, default='./ckpts')
    
    
    parser.add_argument('--fixseed', type=str2bool,default=True)
    parser.add_argument('--seed', type=int, default=97)
    parser.add_argument('--epoch', type=int, default=500)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.name = args.name +'-'+ time.strftime('%Y %m %d-%H %M %S', time.localtime())
    
    args.ckpt_save_path = os.path.join(args.ckpt_save_path, args.name)
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    
    arg_list = args._get_kwargs()
    with open(os.path.join(args.ckpt_save_path, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            print("{:>20}:{:<20}".format(name, arg))
            f.write("{:>20}:{:<20}".format(name, arg)+'\n')
            
    if args.fixseed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    return args

def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    args = get_args()

    # get model, dataloader, optimizer, scheduler,loss function

    train_dataloader, val_dataloader= get_dataloader(args)
    model = get_model(args).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
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
                       f'valBest_{best_test_acc:.3f}_ckpt.pth.tar')

            
        scheduler.step()
        
    f.close()

if __name__ == '__main__':
    main()