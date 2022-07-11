import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/content/homeprices.csv')
df


%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


new_df = df.drop('price',axis='columns')
new_df



price = df.drop('area',axis='columns')
price


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)




#Predict price of a home with area = 3300 sqr ft
reg.predict([[3300]])


reg.coef_


reg.intercept_

plt.xlabel('area',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')


mean_squared_error(df['price'],reg.predict(df[['area']]))

df.price

df1 = pd.read_csv('/content/canada_per_capita_income.csv')
df1

df1 = df1.rename({"per capita income (US$)":"capita"}, axis='columns')

year1 = df1.drop('capita',axis='columns')
year1

capita1 = df1.capita
capita1

# Create linear regression object
reg1 = linear_model.LinearRegression()
reg1.fit(year1,capita1)

reg1.predict([[2020]])

%matplotlib inline
plt.xlabel('year',fontsize=20)
plt.ylabel('percapita',fontsize=20)
plt.scatter(df1.year,df1.capita,color='red',marker='+')
plt.plot(df1.year,reg1.predict(df1[['year']]),color='blue')

