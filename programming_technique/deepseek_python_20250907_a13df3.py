import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def process_data_python(max_obj=None):
    """
    Читает данные в том же формате, что и C++ функция process_data
    """
    with open('./data/data.txt', 'r') as file:
        # Читаем первые два числа: количество образцов и размерность
        n_obj, dim = map(int,file.readline().strip().split())

        # Ограничиваем количество образцов если нужно
        if max_obj is not None:
            n_obj = min(n_obj, max_obj)
        
        # Читаем данные
        X = []
        y = []
        
        for i in range(n_obj):
            line = file.readline().split()
            if len(line) != dim + 1:
                continue
                
            # Читаем признаки
            features = [float(x) for x in line[:dim]]
            # Читаем target
            target = float(line[dim])
            
            X.append(features)
            y.append(target)
    
    # Преобразуем в numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Нормализуем данные (min-max normalization)
    X_max = np.max(np.abs(X), axis=0)
    X_max[X_max == 0] = 1  # избегаем деления на ноль
    X = X / X_max
    
    y_max = np.max(np.abs(y))
    if y_max == 0:
        y_max = 1
    y = y / y_max
    
    return X, y, dim, n_obj

def train_linear_regression(X, y, verbose=True):
    """
    Обучает линейную регрессию и выводит результаты
    """
    # Создаем и обучаем модель
    model = SGDRegressor(
        learning_rate='constant',  # постоянный learning rate
        eta0=0.001,        # значение learning rate
        max_iter=1024,         # количество эпох
        tol=1e-6,                  # tolerance для остановки
        random_state=42,
        verbose=verbose
    )
    model.fit(X, y)
    
    # Предсказания и метрики
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
 
    
    if verbose:
        print("="*60)
        print("Sklearn Linear Regression Results")
        print("="*60)
        print(f"Number of samples: {len(y)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Mean Squared Error: {mse:.6f}")
        
    
    return model, mse



def main():
    
    # Читаем данные (можно изменить размер)
    X, y, dim, n_obj = process_data_python(30000)  # или 512
    

    # Обучаем модель
    model, mse = train_linear_regression(X, y)
    


if __name__ == "__main__":
    main()
