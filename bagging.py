import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    
    df_heart = pd.read_csv('data/Heart_Disease_Dataset.csv')
    print(df_heart['target'].describe())

    X = df_heart.drop(['target'], axis=1)
    y = df_heart['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.35)

    knn_classifier = KNeighborsClassifier().fit(X_train,Y_train)
    knn_pred = knn_classifier.predict(X_test)
    print("="*64)
    print(f'KNN score: {accuracy_score(knn_pred,Y_test)}')


    bag_classifier = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=50).fit(X_train, Y_train)
    bag_pred = bag_classifier.predict(X_test)
    print("="*64)
    print(f'Bagging score: {accuracy_score(bag_pred,Y_test)}')