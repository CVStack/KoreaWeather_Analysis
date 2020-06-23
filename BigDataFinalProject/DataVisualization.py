import csv
import pymongo as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BigDataFinalProject import SoftMaxClassifierClass
from BigDataFinalProject import KNNClassfierClass
from BigDataFinalProject import DataProviderClass

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
    "03": 'spring',
    "04": 'spring',
    "05": 'spring',

    "06": 'summer',
    "07": 'summer',
    "08": 'summer',

    "09": 'fall',
    "10": 'fall',
    "11": 'fall',

    "12": 'winter',
    "01": 'winter',
    "02": 'winter',
}



if __name__ == "__main__":
    data = pd.DataFrame(list(mycol.find()))
    print(data.isnull().sum()) # null 있는지 검사

    dp = DataProviderClass.DataProvider()
    df, df2 = dp.get_TempData(1980)
    #Data Preprocessing

    #NAN Value Remove
    df['_id'] = df['_id'].apply(lambda x: my_dict[x[5:]])  # date -> season
    df2['_id'] = df2['_id'].apply(lambda x: my_dict[x[5:]])  # date -> season
    print(df.isnull().sum())
    print(df)
    print(df['_id'].value_counts())

    df.dropna(axis=0, inplace=True)
    df2.dropna(axis=0, inplace=True)
    #Boxplot visualization and Remove Outlier
    fig, ax = plt.subplots(3,1)

    #과거 데이터
    spring_data = df.groupby('_id').get_group('spring') #dataframe에서 _id가 spring인 요소들 가져옴
    # set_subplot(ax[0], 0, spring_data, 'spring')  # spring_data 표현
    spring_data = removeOutlier(spring_data) #spring_data에서 outlier제거
    # set_subplot(ax[1], 0, spring_data, 'spring') #spring_data 표현

    summer_data = df.groupby('_id').get_group('summer')
    # set_subplot(ax[0], 1, summer_data, 'summer')
    summer_data = removeOutlier(summer_data)
    # set_subplot(ax[1], 1, summer_data, 'summer')

    fall_data = df.groupby('_id').get_group('fall')
    # set_subplot(ax[0], 2, fall_data, 'fall')
    fall_data = removeOutlier(fall_data)
    # set_subplot(ax[1], 2, fall_data, 'fall')

    winter_data = df.groupby('_id').get_group('winter')  # outlier 제거
    # set_subplot(ax[0], 3, winter_data, 'winter')
    winter_data = removeOutlier(winter_data)
    # set_subplot(ax[1], 3, winter_data, 'winter')

    #현재 데이터
    # spring_data = df2.groupby('_id').get_group('spring')  # dataframe에서 _id가 spring인 요소들 가져옴
    # # set_subplot(ax[0], 0, spring_data, 'spring')  # spring_data 표현
    # spring_data = removeOutlier(spring_data)  # spring_data에서 outlier제거
    # # set_subplot(ax[1], 0, spring_data, 'spring') #spring_data 표현
    #
    # summer_data = df2.groupby('_id').get_group('summer')
    # # set_subplot(ax[0], 1, summer_data, 'summer')
    # summer_data = removeOutlier(summer_data)
    # # set_subplot(ax[1], 1, summer_data, 'summer')
    #
    # fall_data = df2.groupby('_id').get_group('fall')
    # # set_subplot(ax[0], 2, fall_data, 'fall')
    # fall_data = removeOutlier(fall_data)
    # # set_subplot(ax[1], 2, fall_data, 'fall')
    #
    # winter_data = df2.groupby('_id').get_group('winter')  # outlier 제거
    # # set_subplot(ax[0], 3, winter_data, 'winter')
    # winter_data = removeOutlier(winter_data)
    # # set_subplot(ax[1], 3, winter_data, 'winter')

    print(spring_data.sort_values('year')) #각 계절에 대해 분포도를 그려야함.
    print(summer_data.sort_values('year'))
    print(fall_data.sort_values('year'))
    print(winter_data.sort_values('year'))

    print('mean')
    print('spring_data : ', 'average - ', np.mean(spring_data['allAverTem']),
          'max - ', np.mean(spring_data['allMaxTem']), 'min - ', np.mean(spring_data['allMinTem']))
    print('summer_data : ', 'average - ', np.mean(summer_data['allAverTem']),
          'max - ', np.mean(summer_data['allMaxTem']), 'min - ', np.mean(summer_data['allMinTem']))
    print('fall_data : ', 'average - ', np.mean(fall_data['allAverTem']),
          'max - ', np.mean(fall_data['allMaxTem']), 'min - ', np.mean(fall_data['allMinTem']))
    print('winter_data : ', 'average - ', np.mean(winter_data['allAverTem']),
          'max - ', np.mean(winter_data['allMaxTem']), 'min - ', np.mean(winter_data['allMinTem']))

    print('median')
    print('spring_data : ', 'average - ', np.median(spring_data['allAverTem']),
          'max - ', np.median(spring_data['allMaxTem']), 'min - ', np.median(spring_data['allMinTem']))
    print('summer_data : ', 'average - ', np.median(summer_data['allAverTem']),
          'max - ', np.median(summer_data['allMaxTem']), 'min - ', np.median(summer_data['allMinTem']))
    print('fall_data : ', 'average - ', np.median(fall_data['allAverTem']),
          'max - ', np.median(fall_data['allMaxTem']), 'min - ', np.median(fall_data['allMinTem']))
    print('winter_data : ', 'average - ', np.median(winter_data['allAverTem']),
          'max - ', np.median(winter_data['allMaxTem']), 'min - ', np.median(winter_data['allMinTem']))


    spring_data = spring_data.sort_values('year')
    fall_data = fall_data.sort_values('year')
    winter_data = winter_data.sort_values('year')
    summer_data = summer_data.sort_values('year')


    averData = pd.concat([spring_data, summer_data ,fall_data, winter_data], axis = 0)

    #histogram
    # ax[0].hist(averData['allAverTem'],100)
    # ax[0].set_xlabel('allAverTem')
    # ax[1].hist(averData['allMaxTem'],100)
    # ax[1].set_xlabel('allMaxTem')
    # ax[2].hist(averData['allMinTem'],100)
    # ax[2].set_xlabel('allMinTem')

    #scatter 가을과 봄의 차이가 없다는 것을 보여줌

    averData['color'] = np.where(averData._id == 'spring', 'red',
                             np.where(averData._id == 'summer', 'blue',
                                      np.where(averData._id == 'fall', 'yellow','gray')))

    # print(averData)
    ax[0].scatter(averData['year'], averData['allAverTem'], c = averData['color'])#aver
    ax[0].set_xlabel('allAverTem')
    ax[1].scatter(averData['year'], averData['allMaxTem'], c = averData['color'])#max
    ax[1].set_xlabel('allMaxTem')
    ax[2].scatter(averData['year'], averData['allMinTem'], c = averData['color'])#min
    ax[2].set_xlabel('allMinTem')
    fig.tight_layout()
    plt.show()


