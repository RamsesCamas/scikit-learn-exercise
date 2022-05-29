import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('data/felicidad.csv')
    print(dataset.head())

    y = dataset[['score']].squeeze()
    X = dataset.drop(['country','score','rank'],axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35,max_iter=10000)
    }

    for name, estimator in estimators.items():
        estimator.fit(X_train,Y_train)
        predictions = estimator.predict(X_test)

        print(name)
        print(f'MSE: {mean_squared_error(Y_test,predictions)}')
        print("Score", estimator.score(X_test, Y_test))