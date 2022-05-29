import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp','family','lifexp','freedom','corruption','generosity','dystopia']]
    Y = dataset[['score']]
    print(X.shape)
    print(Y.shape)

    """
    Load dataset
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

    """
    Make models and predictions
    """
    model_linear_re = LinearRegression().fit(X_train,Y_train)
    y_predict_linear = model_linear_re.predict(X_test)

    #Regularization L1
    model_lasso = Lasso(alpha=0.02).fit(X_train, Y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    #Regularization L2
    model_ridge = Ridge(alpha=1).fit(X_train, Y_train)
    y_predict_ridge = model_ridge.predict(X_test)

    #Regularization Intermedia
    model_elasticnet = ElasticNet(random_state=0).fit(X_train, Y_train)
    y_predict_elasticnet = model_elasticnet.predict(X_test)

    """
    Evaluate the different results
    """

    linear_loss = mean_squared_error(Y_test, y_predict_linear)
    print(f'Linear Loss: {linear_loss}')

    lasso_loss = mean_squared_error(Y_test, y_predict_lasso)
    print(f'Lasso Loss: {lasso_loss}')

    ridge_loss = mean_squared_error(Y_test, y_predict_ridge)
    print(f'Ridge Loss: {ridge_loss}')

    elasticnet_loss = mean_squared_error(Y_test, y_predict_elasticnet)
    print(f'ElasticNet Loss: {elasticnet_loss}')

    """
    View coeficients of the models
    """

    print('='*32)
    print('Coef LASSO')
    print(model_lasso.coef_)

    print('='*32)
    print('Coef Ridge')
    print(model_ridge.coef_)

    print('='*32)
    print('Coef Linear')
    print(model_linear_re.coef_)
    
    print('='*32)
    print('Coef ElasticNet')
    print(model_elasticnet.coef_)