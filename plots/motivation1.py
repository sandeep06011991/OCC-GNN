
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

file_path = "csvs/motivation1.csv"
if file_path.endswith('.csv'):
    data = pd.read_csv(file_path, delimiter=',', header=0)
    data = data.to_records(index=False)


plt.rcParams.update({'font.size': 12})
x = [str(i) for i in data["CachePer"]]
y1 = data["Ratio of Red "]
y2 = data["Time Approx"]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('Cache Percentage')
ax1.set_ylabel('ratio of redundant data', color='g')
ax2.set_ylabel('ratio of data movement time', color='b')

plt.savefig('pngs/motivation1.png')
