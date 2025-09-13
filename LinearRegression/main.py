import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import time
import sys

def process_data_python(data_path='./data/data.txt', max_obj=None):

    with open(data_path, 'r') as file:

        n_obj, dim = map(int,file.readline().strip().split())

        if max_obj is not None:
            n_obj = min(n_obj, max_obj)
   
        X = []
        y = []
        
        for i in range(n_obj):
            line = file.readline().split()
            if len(line) != dim + 1:
                continue
                
       
            features = [float(x) for x in line[:dim]]
      
            target = float(line[dim])
            
            X.append(features)
            y.append(target)
    

    X = np.array(X)
    y = np.array(y)
    
    # Нормализуем данные (min-max normalization)
    X_max = np.max(np.abs(X), axis=0)
    X_max[X_max == 0] = 1  
    X = X / X_max
    
    y_max = np.max(np.abs(y))
    if y_max == 0:
        y_max = 1
    y = y / y_max
    
    return X, y


def train_linear_regression(X, y, n_epochs, initial_lr, decay, verbose=True):

    model = SGDRegressor(
        learning_rate = 'invscaling', 
        eta0 = initial_lr,       
        max_iter = 1,
        shuffle=False,
        power_t = decay,     
        warm_start=True,  
        tol = None,
        penalty = None,                  
        random_state = None,
        verbose = verbose
    )
 
    t0 = time.time()

    model.fit(X, y,
              coef_init=np.random.uniform(-0.1, 0.1, X.shape[1]),
              intercept_init=np.random.uniform(-0.1, 0.1))
    y_pred = model.predict(X)
    mse_max = mean_squared_error(y, y_pred)
    
    for _ in range(n_epochs-1):
         model.fit(X, y)

    elapsed_time = time.time() - t0

    y_pred = model.predict(X)
    mse_min = mean_squared_error(y, y_pred)

    return elapsed_time, mse_max, mse_min



def main():
    args = sys.argv # n_epochs initial_lr data_path
        
    if len(args) == 1:
        n_epochs = 512
        initial_lr = 0.01
        decay_rate = 0.25
        data_path = './data/data.txt'
    elif len(args) == 5:
        n_epochs = int(args[1])
        initial_lr = float(args[2])
        decay_rate = float(args[3])
        data_path = args[4]
    else:
        raise Exception(f"you should pass 0 or 4 arguments, but {len(args)-1} were passed")


    X, y = process_data_python(data_path)  
    elapsed_time, mse_max, mse_min = train_linear_regression(X, y, n_epochs, initial_lr, decay_rate, verbose=False)
    print(elapsed_time, mse_max, mse_min)


if __name__ == "__main__":
    main()