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
#Polinom Reg.
tahminpolinom = PolynomialFeatures(degree=3)
Xyeni = tahminpolinom.fit_transform(x)
polinommodel = LinearRegression()
polinommodel.fit(Xyeni,y)
polinommodel.predict(Xyeni)
plt.plot(x,polinommodel.predict(Xyeni))
plt.show()
hatakaresipolinom = 0

for i in range(len(Xyeni)):
    hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2

    