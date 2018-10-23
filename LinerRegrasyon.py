import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
veri = pd.read_csv("2016dolaralis.csv")
x = veri["Gun"]
y = veri["Fiyat"]
x = x.reshape(251,1)
y= y.reshape(251,1)
plt.scatter(x,y)
plt.show()
#Lineer Reg.
tahminlineer = LinearRegression()
tahminlineer.fit(x,y)
tahminlineer.predict(x)
plt.plot(x,tahminlineer.predict(x),c="red")

hatakaresilineer = 0

for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(tahminlineer.predict(x)[i]))**2
