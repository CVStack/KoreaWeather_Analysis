import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pandas as pd

#SoftMax regression을 실행하는 클래스
#model 생성, 결과값 시각화, 결과값 출력
class KNNClassifier:

    def __init__(self, training_data, training_label, n): #생성자 -> 훈련 데이터와 이에 대한 결과값을 load
        self.training_data, self.test_data, self.training_label, self.test_label = \
            train_test_split(training_data, training_label, test_size=0.2, random_state=4)
        self.knnc = KNeighborsClassifier(n_neighbors=n)
        self.scaler = RobustScaler()  # normalization
            #자동으로 테스트 데이터 생성

    def train(self):
        if len(self.training_data) == 0 and len(self.training_label) == 0 :
            return
        # print(self.training_data)
        self.scaler.fit(self.training_data)
        # # print(self.scaler.mean_)
        self.training_data = self.scaler.transform(self.training_data)
        self.test_data = self.scaler.transform(self.test_data)
        # print(self.training_data)

        self.knnc.fit(self.training_data, self.training_label)

    def test(self):
        if len(self.test_data) == 0 :
            return
        result = self.knnc.predict(self.test_data)
        # print('test_label')
        # print(self.test_label)
        # print('model_test_label')
        # print(result)

        # print(metrics.accuracy_score(result, self.test_label))
        confusion_matrix_rs = confusion_matrix(self.test_label, result)
        metrics_rs = metrics.classification_report(self.test_label, result, digits=3)
        return [metrics.accuracy_score(result, self.test_label), confusion_matrix_rs, metrics_rs];

    def experiment(self, exper_point, exper_labels):

        print(exper_point)
        result = self.knnc.predict(self.scaler.transform(exper_point.iloc[0:, 2:]))
        # print(result)

        df_rs = pd.DataFrame({
            'year' : exper_point['year'],
            'month' : exper_point['month'],
            'result' : result
        })
        #result 결과를 담는 dataframe 생성 후 return
        return df_rs

if __name__ == "__main__":
    iris_data = datasets.load_iris()
    sc = KNNClassifier(iris_data.data, iris_data.target)

    sc.train()
    sc.test()