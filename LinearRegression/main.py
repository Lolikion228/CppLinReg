import os
from tqdm import tqdm
from lr_exec import process_data
from sklearn.decomposition import PCA
import numpy as np


def perform_pca():

    X, y = process_data('./data/orig_data.txt')

    dim = X.shape[1]

    for d in tqdm(range(1, dim+1), desc='doing pca...'):
        pca = PCA(d)
        X_new = pca.fit_transform(X)
        f = open(f"./data/data_{d}.txt", 'a')
        f.write(f"{X.shape[0]} {d}\n")

        for i in range(X.shape[0]):
            f.write( str(X_new[i,0]) )
            for j in range(1,d):
                f.write( ' ' +  str(X_new[i,j]) )
            f.write(' ' + str(y[i])+'\n')
        f.close()


def grid_search():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    if os.path.exists('logs/py_log.txt'):
        os.remove('./logs/py_log.txt')

    if os.path.exists('logs/cpp_log.txt'):
        os.remove('./logs/cpp_log.txt')


    n_epochs = [128, 256, 512, 1024, 2048, 4096][:2]
    initial_lr = [1.0, 1e-1, 1e-2, 1e-3, 1e-4][:2]
    decay_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7][:2]
    data_files = [ x for x in os.listdir('./data') if x.startswith('data')][:2]

    pbar = tqdm(total= len(n_epochs) * len(initial_lr) * len(decay_rate) * len(data_files), desc='grid search')

    for ne in n_epochs:
        for il in initial_lr:
            for dr in decay_rate:
                for df in data_files:

                    py_f = open('./logs/py_log.txt', 'a')
                    py_f.write(f"{ne} {il} {dr} {df} ")
                    py_f.close()
                    os.system(f"python ./lr_exec.py {ne} {il} {dr} {'./data/' + df}  >> ./logs/py_log.txt")

                    c_f = open('./logs/cpp_log.txt', 'a')
                    c_f.write(f"{ne} {il} {dr} {df} ")
                    c_f.close()
                    os.system(f"./lr_exec {ne} {il} {dr} {'./data/' + df} >> ./logs/cpp_log.txt")

                    pbar.update()

def main():
    # perform_pca()
    grid_search()

if __name__ == "__main__":
    main()

