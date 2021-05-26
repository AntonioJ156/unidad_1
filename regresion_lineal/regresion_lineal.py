import pandas as pd
import numpy as np
from sklearn import linear_model

dep_df = pd.read_csv('departamentos-pro.csv')
house_df = pd.read_csv('casas-pro.csv')

dep_df.fillna(0)
dep_model = linear_model.LinearRegression()

dep_refugio = dep_df.loc[dep_df['web-scraper-start-url'].str.contains("el-refugio")]
len = int(dep_refugio['precio'].size * 0.7)

dep_70 = dep_refugio.iloc[0:len]

dep_model.fit(pd.DataFrame(dep_70['mets']), pd.DataFrame(dep_70['precio']))
dep_model.predict(pd.DataFrame([125]))

dep_70_100 = dep_refugio.iloc[len:]

dep_model.fit(pd.DataFrame(dep_70_100['mets']), pd.DataFrame(dep_70_100['precio']))
dep_model.predict(pd.DataFrame([125]))

house_df.fillna(0)
model_house = linear_model.LinearRegression()

casas_refugio = house_df.loc[house_df['web-scraper-start-url'].str.contains("refugio")]
len = int(casas_refugio['precio'].size * 0.7)

house_70 = casas_refugio.iloc[0:len]

model_house.fit(pd.DataFrame(house_70['mets']), pd.DataFrame(house_70['precio']))
model_house.predict(pd.DataFrame([492]))

house_70_100 = pd.DataFrame(casas_refugio.iloc[len:])

model_house.fit(pd.DataFrame(house_70_100['mets']), pd.DataFrame(house_70_100['precio']))
model_house.predict(pd.DataFrame([492]))