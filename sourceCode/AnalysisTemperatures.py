import csv
import pymongo as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BigDataFinalProject import SoftMaxClassifierClass
from BigDataFinalProject import KNNClassfierClass
from BigDataFinalProject import DataProviderClass
from scipy import stats

from sklearn.preprocessing import StandardScaler

myclient = pm.MongoClient()
mydb = myclient["AnalysisTemperature"]
mycol = mydb["AllTemperature2"]


# main 소스코드

# 함수
def removeOutlier(data): #outlier를 제거하는 함수
    feature_list = ['allAverTem', 'allMaxTem', 'allMinTem'] #이 feature에 대해 이상치를 제거함
    filter_list = []

    for item in feature_list: #boxplot 공식
        Q1 = data[item].quantile(0.25)
        Q3 = data[item].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filter = (data[item] >= Q1 - 1.5 * IQR) & (data[item] <= Q3 + 1.5 * IQR) #bolxplot 조건
        filter_list.append(filter)

    # print(filter_list[0] and filter_list[1] and filter_list[2])
    data = data.loc[filter_list[0]] #filter : allAverTem 의 조건을 만족시키는 요소들만 남김
    data = data.loc[filter_list[1]] #filter : allMaxTem 의 조건을 만족시키는 요소들만 남김
    data = data.loc[filter_list[2]] #filter : allMinTem 의 조건을 만족시키는 요소들만 남김
    return data #이상치 제거한 데이터 리턴

def set_subplot(ax, index, data, season) :
    ax[index].boxplot([data['allAverTem'], data['allMaxTem'], data['allMinTem']])
    ax[index].set_title(season)
    ax[index].set_xticklabels(['AverTem', 'MaxTem', 'MinTem'])
    ax[index].set_ylabel('value')

# data preprocessing --> NAM 제거, outlier 값 제거

my_dict = {  # 0 : spring, 1 : summer, 2 : fall, 3 : winter
    "03": 'spring/fall',
    "04": 'spring/fall',
    "05": 'spring/fall',

    "06": 'summer',
    "07": 'summer',
    "08": 'summer',

    "09": 'spring/fall',
    "10": 'spring/fall',
    "11": 'spring/fall',

    "12": 'winter',
    "01": 'winter',
    "02": 'winter',
}



if __name__ == "__main__":
    past = 1980
    data = pd.DataFrame(list(mycol.find()))
    # print(data.isnull().sum()) # null 있는지 검사

    dp = DataProviderClass.DataProvider()
    df, df2 = dp.get_TempData(past)
    #Data Preprocessing

    #NAN Value Remove
    df['_id'] = df['_id'].apply(lambda x: my_dict[x[5:]])  # date -> season

    # print(df.isnull().sum())
    # print(df)
    # print(df['_id'].value_counts())

    df.dropna(axis=0, inplace=True)

    #Boxplot visualization and Remove Outlier
    fig, ax = plt.subplots(2, 3)

    spring_fall_data = df.groupby('_id').get_group('spring/fall') #dataframe에서 _id가 spring인 요소들 가져옴
    set_subplot(ax[0], 0, spring_fall_data, 'spring/fall')  # spring_data 표현
    spring_fall_data = removeOutlier(spring_fall_data) #spring_data에서 outlier제거
    set_subplot(ax[1], 0, spring_fall_data, 'spring/fall') #spring_data 표현

    # spring_data = df.groupby('_id').get_group('spring') #dataframe에서 _id가 spring인 요소들 가져옴
    # set_subplot(ax[0], 0, spring_data, 'spring')  # spring_data 표현
    # spring_data = removeOutlier(spring_data) #spring_data에서 outlier제거
    # set_subplot(ax[1], 0, spring_data, 'spring') #spring_data 표현

    summer_data = df.groupby('_id').get_group('summer')
    set_subplot(ax[0], 1, summer_data, 'summer')
    summer_data = removeOutlier(summer_data)
    set_subplot(ax[1], 1, summer_data, 'summer')

    # fall_data = df.groupby('_id').get_group('fall')
    # set_subplot(ax[0], 2, fall_data, 'fall')
    # fall_data = removeOutlier(fall_data)
    # set_subplot(ax[1], 2, fall_data, 'fall')

    winter_data = df.groupby('_id').get_group('winter')  # outlier 제거
    set_subplot(ax[0], 2, winter_data, 'winter')
    winter_data = removeOutlier(winter_data)
    set_subplot(ax[1], 2, winter_data, 'winter')

    plt.show()

    #trainging

    # AllSeason_data = pd.concat([spring_data.iloc[0: int(len(spring_data) / 2)], summer_data, fall_data.iloc[0: int(len(fall_data) / 2)], winter_data], axis = 0)
    AllSeason_data = pd.concat([spring_fall_data.iloc[0: int(len(spring_fall_data) / 2)], summer_data, winter_data], axis = 0) #fall이랑 spring 이랑 비슷한 기후여서 한개 뻄
    # AllSeason_data = pd.concat([spring_fall_data, summer_data, winter_data], axis=0)
    # print(AllSeason_data)
    print(AllSeason_data['_id'].value_counts())
    print(AllSeason_data.sort_values('year')['year'].value_counts())

    trainging_points = AllSeason_data.iloc[0:, 3:]
    trainging_labels = AllSeason_data['_id']

    # print(trainging_points)
    # print(trainging_labels)

    sc = SoftMaxClassifierClass.SoftmaxClassifier(trainging_points, trainging_labels.ravel())
    sc.train()
    accuracy_result = sc.test()

    for x in accuracy_result: #accurancy를 출력
        print(x)

    accuracy_scores = []

    for x in range(1,201):
        kc = KNNClassfierClass.KNNClassifier(trainging_points, trainging_labels, x)
        kc.train()
        accuracy_scores.append(kc.test()[0])

    print('KNN', max(accuracy_scores))
    plt.plot(range(1, 201), accuracy_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

    #experiment
    max_k = accuracy_scores.index(max(accuracy_scores)) + 1 #가장 높은 정확도를 보인 k값을 가져옴
    print('max_k ',max_k)
    kc = KNNClassfierClass.KNNClassifier(trainging_points, trainging_labels, max_k)
    kc.train()

    accuracy_result = kc.test()

    for x in accuracy_result: #accurancy를 출력
        print(x)

    # print(df2) #current data 출력
    print('knn')
    result =kc.experiment(df2.iloc[0:, 1:], df2['_id'])
    print(result)
    result = result[result['year'] == 2017]
    print(result)

    print('softmax')
    result = sc.experiment(df2.iloc[0:, 1:])
    print(result)
    print(result['result'].value_counts())
    print(stats.mode(result['result']))

    total_year_result = result['result'].value_counts()
    total_year_result.plot.pie(autopct='%.2f%%')
    plt.show()

    fig, ax = plt.subplots(5,8)
    x = 0
    y = 0
    for year in range(1981, 2020):
        year_rs = result[result['year'] == year]['result'].value_counts()
        ax[y,x].pie(year_rs, labels= year_rs.index ,autopct='%1.1f%%', shadow=True)
        ax[y,x].set_xlabel(year)
        x +=1
        if x == 8:
            x = 0
            y += 1

    fig.tight_layout()
    plt.show()
