import time
import os
import pickle
import numpy as np
from argparse import ArgumentParser

root_dir_name = 'data'

def main(size_per_label = 10000, train_test_ratio = 4, verbose=True):
    print(">>>>开始游戏！>>>>")
    data_dict = {
        'left': [],
        'right': [],
        'up': [],
        'down': [],
        'empty': [],
    }
    target = input('想要生成哪类数据呢^^ (left/right/up/down/empty/all)')
    while target != 'all' and target not in data_dict.keys():
        target = input("请检查拼写(left/right/up/down/empty/all)").strip()
        print(target)
    if target == 'all':
        print("即将启动完整数据生成程序")
        target = None
    else:
        print(f"即将开始生成'{target}'数据")
    for instruction in data_dict.keys():
        if target == None or instruction == target:
            try:
                cnt = 0
                while cnt < size_per_label:
                    print(">>>就绪>>>")
                    time.sleep(1)

                    print(instruction)
                    data = collecte_data()
                    need_save = ask_if_save(verbose)
                    if need_save:
                        data_dict[instruction].extend(data)
                        cnt += 1
                        print(">>>采集成功>>>")
                    else:
                        print(">>>丢弃>>>")
                    time.sleep(1)

                c = input("主人需要休息一下吗？按Q退出，之前的数据都会帮您存好的！不需要休息就按其他任意键继续吧\n").strip()
                if c.capitalize() == 'Q':
                    do_save(data_dict, train_test_ratio)
                    print("保存成功！")
                    return
                print("马上要换动作啦，做好准备哦^^")
                time.sleep(3)
            except BaseException as e:
                do_save(data_dict, train_test_ratio)
                print("呃啊，有异常发生了，但我临死之前为主人存好了所有数据，别为我哭泣~~")
                raise e
    do_save(data_dict)

def do_save(data_dict:dict, train_test_ratio):
    os.makedirs(root_dir_name, exist_ok=True)
    for instruction, data in data_dict.items():
        if len(data) > 0:
            split = int(len(data) * train_test_ratio / (train_test_ratio + 1))
            train_data = np.array(data[:split])
            test_data = np.array(data[split:])
            train_dir = os.path.join(root_dir_name, 'train', instruction)
            test_dir = os.path.join(root_dir_name, 'test', instruction)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            train_path = time.strftime('%Y %m %d-%H %M %S', time.localtime()) + '.npy'
            train_path = os.path.join(train_dir, train_path)
            test_path = time.strftime('%Y %m %d-%H %M %S', time.localtime()) + '.npy'
            test_path = os.path.join(test_dir, test_path)
            with open(train_path, 'wb') as f:
                np.save(f, train_data)
            with open(test_path, 'wb') as f:
                np.save(f, test_data)
    print("保存成功")

def collecte_data():
    time.sleep(0.5)
    return [1]

def ask_if_save(verbose):
    if verbose:
        c = input("是否采集呢？(Y/N)")
        if c.capitalize() == 'Y':
            return True
        else:
            return False
    return False

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--size_per_label', type=int, default=10)
    args, _ = parser.parse_known_args()
    main(size_per_label=args.size_per_label, verbose=args.verbose, train_test_ratio=4)
            
            
            
            


