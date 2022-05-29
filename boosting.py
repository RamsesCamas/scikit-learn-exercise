import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    
    df_heart = pd.read_csv('data/Heart_Disease_Dataset.csv')
    print(df_heart['target'].describe())

    X = df_heart.drop(['target'], axis=1)
    y = df_heart['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=100,loss='exponential',learning_rate=0.15, max_depth=5).fit(X_train,Y_train)
    boost_pred = boost.predict(X_test)

    print('='*64)
    print(f'Boosting score {accuracy_score(boost_pred,Y_test)}')