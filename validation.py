import torch
from utils import  AverageMeter, accuracy
from dataloader import  get_dataloader
from model import get_model
from arguments import get_args
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    args = get_args()

    _, val_dataloader= get_dataloader(args)
    model = get_model(args).cuda()
    model.load_state_dict(torch.load(args.ckpt_load_path)['state_dict'])
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    val_acc = 0.0

    val_acc_recoder = AverageMeter()
    val_loss_recoder = AverageMeter()
        
    val_loss_recoder.reset()
    val_acc_recoder.reset()
    
    '''
        Validation!
    '''
    preds = []
    truths = []
    
    model.eval()
    with torch.no_grad():
        for input, action in tqdm(val_dataloader):
            input,action = input.cuda(),action.cuda()

            prediction = model(input)
            val_loss = loss_function(prediction, action)
            val_acc = accuracy(prediction, action)

            val_loss_recoder.update(val_loss.item(), n=action.size(0))
            val_acc_recoder.update(val_acc.item(), n=action.size(0))
            
            pred_label = torch.argmax(prediction, dim=1)
            preds += list(map(int, pred_label))
            truths += list(map(int, action))
            
    '''
        Logging!
    '''
    
    msg = f'Val loss,{val_loss_recoder.avg:.4f},val acc,{val_acc_recoder.avg:.2f}'
    
    print(msg)
    
    cfx_matrix = confusion_matrix(truths, preds)
    draw_acc_and_confusion_matrix(cfx_matrix)

def draw_acc_and_confusion_matrix(cfx_mtx):
    # global args

    # labels = ['up','down','left','right','empty']
    labels = ['left','right','empty']
    plt.figure(figsize=(6,5))
    plt.imshow(cfx_mtx)
    plt.colorbar()

    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(labels)), labels,rotation=90)#X轴字体倾斜45°
    plt.subplots_adjust(top=0.97,
        bottom=0.31,
        left=0.146,
        right=0.975,
        hspace=0.2,
        wspace=0.2)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()