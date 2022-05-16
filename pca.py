import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_heart = pd.read_csv('./data/Heart_Disease_Dataset.csv')

    print(df_heart)

    df_features = df_heart.drop(['target'],axis=1)
    df_target = df_heart['target']


    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, Y_train, Y_test = train_test_split(df_features,df_target,test_size=0.3,random_state=42)

    print(X_train.shape)
    print(Y_train.shape)

    #n_components = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)

    logistic.fit(df_train, Y_train)

    print("SCORE PCA: ",logistic.score(df_test, Y_test))

    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)

    logistic.fit(df_train, Y_train)

    print("SCORE IPCA: ",logistic.score(df_test, Y_test))