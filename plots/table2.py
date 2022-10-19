import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def load_data(file_path: str):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, delimiter=',', header=0)
        data = data.to_records(index=False)
        return data

# create data
x = [1,2,3,4,5]
y = [3,3,3,3,3]

# plot lines
plt.plot(x, y, label = "line 1")
plt.plot(y, x, label = "line 2")
plt.plot(x, np.sin(x), label = "curve 1")
plt.plot(x, np.cos(x), label = "curve 2")
plt.legend()
plt.show()
fig, ax = plt.subplots(10, 10)

if __name__=="__main__":
