import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from numpy.ma.core import reshape
from sklearn.linear_model import LinearRegression

df = pd.read_csv("House.csv")

x = df["metraj"]
y = df["price"]

X = np.array(x).reshape(-1,1)

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

print("Coefficient:",model.coef_[0])
print("Intercept:",model.intercept_)
print("R² Score:",model.score(X,y))

prediction = model.predict([[90]])

print("metraj 90 is:", prediction[0])

plt.figure(figsize=(8,8))
plt.scatter(x,y,label="Actual Data")
plt.plot(X,y_pred,label="Regression Line")
plt.xlabel("Metraj")
plt.ylabel("Price")
plt.legend()
plt.show()
