from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
datas = "laptop_battery_prediction/data.txt"
with open(datas, 'r') as f:
    lines = f.readlines()
data = [list(map(float, line.strip().split(','))) for line in lines]
df = pd.DataFrame(data, columns=['UsageTime', 'BatteryHealth'])
x = [[i] for i in df.UsageTime]
y = [[i] for i in df.BatteryHealth]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
x_test = float(input("Enter the usage time of laptop in hours:"))
print("The predicted battery health is:", model.predict(np.array([x_test]).reshape(-1, 1)))