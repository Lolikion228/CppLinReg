import os
from tqdm import tqdm
import time



def grid_search():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    if os.path.exists('logs/py_log.txt'):
        os.remove('./logs/py_log.txt')

    if os.path.exists('logs/cpp_log.txt'):
        os.remove('./logs/cpp_log.txt')


    n_epochs = [128, 256, 512, 1024, 2048, 4096][:2]
    initial_lr = [1.0, 1e-1, 1e-2, 1e-3, 1e-4][:3]
    decay_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7][:3]
    dp = './data/data.txt'

    pbar = tqdm(total= len(n_epochs) * len(initial_lr) * len(decay_rate), desc='grid search')

    for ne in n_epochs:
        for il in initial_lr:
            for dr in decay_rate:

                py_f = open('./logs/py_log.txt', 'a')
                py_f.write(f"{ne} {il} {dr} ")
                py_f.close()
                os.system(f"python ./lr_exec.py {ne} {il} {dr} {dp}  >> ./logs/py_log.txt")

                c_f = open('./logs/cpp_log.txt', 'a')
                c_f.write(f"{ne} {il} {dr} ")
                c_f.close()
                os.system(f"./lr_exec {ne} {il} {dr} {dp} >> ./logs/cpp_log.txt")

                pbar.update()

def main():
    grid_search()

if __name__ == "__main__":
    main()

