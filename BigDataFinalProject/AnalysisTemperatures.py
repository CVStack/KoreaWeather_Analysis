import csv
import pymongo as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    ax[index].set_xlabel('attr')
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

# data = pd.DataFrame(list(mycol.find()))
# print(data.isnull().sum()) # null 있는지 검사

pipeline = [
    {"$group": {"_id": "$date", "allAverTem": {"$avg": "$averTem"}, "allMaxTem": {"$avg": "$MaxTem"},
                "allMinTem": {"$avg": "$MinTem"}}}
]

print(len(list(mycol.aggregate(pipeline))))

df = pd.DataFrame(list(mycol.aggregate(pipeline))) #결과 dataframe에 담음

df['_id'] = df['_id'].apply(lambda x: my_dict[x[5:]])  # date -> season

print(df.isnull().sum())
print(df)

df.dropna(axis = 0, inplace=True) #NAN이 있는 행 delete

# print(df.count_values())



fig, ax = plt.subplots(1, 4)

spring_data = df.groupby('_id').get_group('spring') #dataframe에서 _id가 spring인 요소들 가져옴
spring_data = removeOutlier(spring_data) #spring_data에서 outlier제거
set_subplot(ax, 0, spring_data, 'spring') #spring_data 표현
#
# summer_data = df.groupby('_id').get_group('summer')
# summer_data = removeOutlier(summer_data)
# set_subplot(ax, 1, summer_data, 'summer')
#
# fall_data = df.groupby('_id').get_group('fall')
# fall_data = removeOutlier(fall_data)
# set_subplot(ax, 2, fall_data, 'fall')
#
# winter_data = df.groupby('_id').get_group('winter')  # outlier 제거
# winter_data = removeOutlier(winter_data)
# set_subplot(ax, 3, winter_data, 'winter')
#
# plt.show()

# data visualization

# pipeline = [
#     {"$project" : {"averTem" : 1, "date" : 1,"_id" : 0}}
# ]
# df = pd.DataFrame(list(mycol.aggregate(pipeline))) #해당 파이프 라인으로 데이터를 얻어옴
#
# df['date'] = df['date'].apply(lambda x : x[:4]) #date columns을 가져와서 indexing --> 1972-01 --> 1972
# df['averTem'] = df['averTem'].apply(pd.to_numeric) #averTem column 전부에 to_numeric 함수 적용
#
# gp = df.groupby('date') #date로 그룹핑
# averTemp_perYear = df.groupby('date').mean() #그룹핑한 후 평균값을 얻음 -> ~~년대 평균을 얻을수잇음
# # print(averTemp_perYear)
#
# x_labels = list(gp.groups.keys())
# y_labels = averTemp_perYear['averTem']
# # print(y_labels.values)
# # print(gp.groups.keys())
#
# plt.plot(x_labels, y_labels)
# plt.show()


