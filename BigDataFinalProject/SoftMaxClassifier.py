import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#SoftMax regression을 실행하는 클래스
#model 생성, 결과값 시각화, 결과값 출력
class SoftmaxClassifier:
    softreg = LogisticRegression(C=1e5) # logistic model을 생성, C = 정규화 수치

    def __init__(self, training_data, training_label): #생성자 -> 훈련 데이터와 이에 대한 결과값을 load
        self.training_data, self.test_data, self.training_label, self.test_label = \
            train_test_split(training_data, training_label, test_size=0.2, random_state=4)
        self.scaler = StandardScaler()  # normalization
            #자동으로 테스트 데이터 생성

    # def __init__(self, training_data, training_label, test_data, test_label): #생성자 -> 훈련 데이터와 이에 대한 결과값을 load
    #     self.training_data = training_data
    #     self.training_label = training_label
    #     self.test_data = test_data
    #     self.test_label = test_label

    def train(self):
        if len(self.training_data) == 0 and len(self.training_label) == 0 :
            return
        self.scaler.fit(self.training_data) # compute mean and std

        self.softreg.fit(self.scaler.transform(self.training_data), self.training_label)

    def test(self):
        if len(self.test_data) == 0 :
            return
        result = self.softreg.predict(self.scaler.transform(self.test_data))
        print('test_label')
        print(self.test_label)
        print('model_test_label')
        print(result)

        print(metrics.accuracy_score(result, self.test_label))

    # def test(self, test_data):
    #     self.test_data = test_data
    #     self.test();

# Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

if __name__ == "__main__":
    iris_data = datasets.load_iris()
    sc = SoftmaxClassifier(iris_data.data, iris_data.target)

    sc.train()
    sc.test()