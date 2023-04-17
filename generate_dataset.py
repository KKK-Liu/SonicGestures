import time
import os
import pickle
import numpy as np
from argparse import ArgumentParser
import pygame

ROOT_DIR_NAME = 'data'
IMG_DIR_NAME = 'imgs'

def str_play(size_per_label = 10000, train_test_ratio = 4, verbose=True):
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
                print("呃啊，有异常发生了，但我临死之前为主人存好了所有数据，请不用担心~~")
                raise e
    do_save(data_dict)

def ask_if_save(verbose):
    if verbose:
        c = input("是否采集呢？(Y/N)")
        if c.capitalize() == 'Y':
            return True
        else:
            return False
    return False

def UI_play(train_test_ratio=4):

    os.makedirs(ROOT_DIR_NAME, exist_ok=True)
    pygame.init()
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    class Prompt(pygame.sprite.Sprite):

        def __init__(self, img_path) -> None:
            super().__init__()
            self.surface = pygame.image.load(os.path.join(IMG_DIR_NAME, img_path)).convert()
            self.surface.set_colorkey((255, 255, 255), pygame.RLEACCEL)
            self.rect = self.surface.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        

    # Just for test, can be removed
    def test_save(data_dict, train_test_ratio):
        with open(os.path.join('data', 'data.txt'), 'w') as f:
            for key, value in data_dict.items():
                f.write(f'{key}: {value}\n')
    
    running = True
    instruction = None
    key_map = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right', ' ': 'empty'}
    data_dict = {'up': [], 'left': [], 'down': [], 'right': [], 'empty': []}
    prompts = {'down': Prompt('down.png'), 'up': Prompt('up.png'), 'left': Prompt('left.png'), 'right': Prompt('right.png'), 'empty': Prompt('empty.png')}
    pygame.event.set_blocked(None)
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
    cnt = 0
    screen.fill((255, 255, 255))
    while running:
        lst = pygame.event.get()
        for e in lst:
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif instruction is None and e.unicode in key_map.keys():
                    instruction = key_map[e.unicode]
                    prompt = prompts[key_map[e.unicode]]
                    screen.blit(prompt.surface, prompt.rect)
                    collection = []
            elif e.type == pygame.KEYUP:
                if e.unicode in key_map.keys() and key_map[e.unicode] == instruction:
                    collection = clean(np.array(collection))
                    data_dict[instruction].append(collection)
                    screen.fill((255, 255, 255))
                    print(f'{key_map[e.unicode]} {len(collection)} successfully collected!')
                    '''
                    Maybe we can save current data as a tmp file here to prevent unexpected crash?
                    '''
                    instruction = None
                    cnt = 0
        if instruction is not None:
            data = collecte_data()
            collection.append(data)
            foo(100) # remember to delete this line!
            cnt += 1
        pygame.display.flip()

    do_save(data_dict, train_test_ratio)
    pygame.quit()

def foo(N):
    for i in range(N):
        with open(os.path.join('data', 'test.txt'), 'w') as f:
            print('foo', file=f)

def do_save(data_dict:dict, train_test_ratio):
    os.makedirs(ROOT_DIR_NAME, exist_ok=True)
    for instruction, data in data_dict.items():
        if len(data) > 0:
            data = np.concatenate(data, axis=0)
            data = np.random.permutation(data)
            split = int(len(data) * train_test_ratio / (train_test_ratio + 1))
            train_data = np.array(data[:split])
            test_data = np.array(data[split:])
            train_dir = os.path.join(ROOT_DIR_NAME, 'train', instruction)
            test_dir = os.path.join(ROOT_DIR_NAME, 'test', instruction)
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
    print(">>>>>Successfully Saved>>>>>>>")

# TODO: receive data in one time slice from arduino
def collecte_data(T=50):
    '''
    Return np.ndarray of shape (T, 5, 5)
    '''
    return np.random.randn(T, 5, 5)

# TODO: data cleaning, better ideas?
def clean(data: np.ndarray, thres = 0):
    tmp = data.reshape((data.shape[0], -1))
    # drop first
    mask = np.abs(tmp[1:] - tmp[:-1]).mean(axis=1) > thres
    return data[1:][mask]





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--size_per_label', type=int, default=10)
    args, _ = parser.parse_known_args()
    # main(size_per_label=args.size_per_label, verbose=args.verbose, train_test_ratio=4)
    UI_play()
            
            
            
            


