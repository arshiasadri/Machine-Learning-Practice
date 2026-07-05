import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("Plant.csv")

x = df["Day"]
y = df["High"]

order = np.argsort(x)

X = np.array(x).reshape(-1,1)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly,y)

y_pred = model.predict(X_poly)

print("Coefficient:",model.coef_[0])
print("Intercept:",model.intercept_)
print("R² Score:",model.score(X_poly,y))

prediction = model.predict(poly.transform([[7]]))

print("High in day 7 is:", prediction[0])

plt.figure(figsize=(8,8))
plt.scatter(x,y,label = "Actual Data")
plt.plot(x.iloc[order], y_pred[order], label="Polynomial Regression")
plt.xlabel("Day")
plt.ylabel("High")
plt.legend()
plt.show()