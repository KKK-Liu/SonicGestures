import argparse
import os
import time
import torch
import numpy as np
import random

from utils import str2bool

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='FCLayer')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    
    ''' dataloader '''
    parser.add_argument('--data_root', type=str, default='./data-5')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    
    ''' model and network '''
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)    
    parser.add_argument('--weight_decay', type=float, default=1e-5)    
    parser.add_argument('--milestones', type=int, nargs='+',default=[100,200,300, 500, 800])
    parser.add_argument('--gamma', type=float,default=0.1)


    parser.add_argument('--ckpt_save_path', type=str, default='./ckpts')
    parser.add_argument('--ckpt_load_path', type=str, default=r'D:\vscodefile\SonicGestures\ckpts\name-2023 05 19-21 44 51\valBest_77.722_ckpt.pth.tar')
    
    parser.add_argument('--fixseed', type=str2bool,default=True)
    parser.add_argument('--seed', type=int, default=97)
    
    
    parser.add_argument('--port', type=str, default='COM5')
    parser.add_argument('--baud', type=int, default=115200)
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