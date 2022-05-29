import pandas as pd
from sklearn.cluster import MeanShift


if __name__ == '__main__':
    dataset = pd.read_csv('data/candy-data.csv')
    print(dataset.head(5))

    X = dataset.drop('competitorname',axis=1)

    mean_shift = MeanShift().fit(X)
    print(mean_shift.labels_)
    print(max(mean_shift.labels_))
    print('='*64)
    print(mean_shift.cluster_centers_)

    dataset['meanshift'] = mean_shift.labels_

    print('='*64)
    print(dataset.head(5))