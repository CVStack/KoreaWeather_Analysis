import pymongo as pm
import pandas as pd

myclient = pm.MongoClient()
mydb = myclient["AnalysisTemperature"]
mycol = mydb["AllTemperature2"]

class DataProvider:

    def __init__(self): #mongodb에 들어 있는 값을 reformating month field와 year field 추가
        self.myclient = pm.MongoClient()
        self.mydb = myclient["AnalysisTemperature"]
        self.mycol = mydb["AllTemperature2"]
        # pipeline = [
        #     {"$group": {"_id": "$date"}}
        # ]
        # for item in list(mycol.aggregate(pipeline)):
        #     date = item['_id']
        #
        #     year = int(date[:4])
        #     month = int(date[5:])
        #
        #     filter = {
        #         'date': date
        #     }
        #     newContent = {
        #         'year': year,
        #         'month': month
        #     }
        #     mycol.update_many(filter, {'$set': newContent})

    def get_TempData(self, n): #과거 데이터, 현재 데이터를 나눠서 제공함 n을 기준으로 과거, 현재로 리턴

        pipeline = [{}, {}]

        pipeline[1] = {"$group": {"_id": "$date", 'year' : {'$first' : '$year'}, 'month' : {'$first' : '$month'}, "allAverTem": {"$avg": "$averTem"}, "allMaxTem": {"$avg": "$MaxTem"},
                        "allMinTem": {"$avg": "$MinTem"}}}
        pipeline[0] = {"$match": {"year": {"$lte": n}}}
        past_DataFrame = pd.DataFrame(list(self.mycol.aggregate(pipeline)))
        pipeline[0] = {"$match": {"year": {"$gt": n}}}
        current_DataFrame = pd.DataFrame(list(self.mycol.aggregate(pipeline)))

        return past_DataFrame, current_DataFrame


if __name__ == '__main__':

    dp = DataProvider()

    past, current = dp.get_TempData(1990)

    print(past)
    print(current)