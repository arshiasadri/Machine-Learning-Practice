import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Student_time.csv")

x = df['Time']
y = df['Score']


X = np.array(x).reshape(-1, 1)


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", model.score(X, y))


prediction = model.predict([[7]])
print("Predicted score for 7 hours:", prediction[0])


plt.figure(figsize=(8, 8))
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, label="Regression Line")
plt.xlabel("Time")
plt.ylabel("Score")
plt.title("Student Study Time vs Score")
plt.legend()
plt.show()
