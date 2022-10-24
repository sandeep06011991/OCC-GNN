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


# plot lines
def dummy_plot():
    # x = [1,2,3,4,5]
    # y = [3,3,3,3,3]
    #
    # # plot lines
    # plt.plot(x, y, label = "line 1")
    # plt.plot(y, x, label = "line 2")
    # plt.plot(x, np.sin(x), label = "curve 1")
    # plt.plot(x, np.cos(x), label = "curve 2")
    # plt.legend()
    # plt.savefig('full_figure.png')
    # print("done")
    # return
    fig, ax = plt.subplots(2, 3)

    cache_per = ["0",".1","25",".5","1"]
    b = [i for i in range(5)]
    models = ["gcn", "gat"]
    datasets = ["ogbn-arxiv", "ogbn-papers", "amazon"]
    for x,m in enumerate(models):
        for y,d in enumerate(datasets):
            print(x,y, cache_per , [i * 3 for i in range(5)] )
            ax[x,y].set_title(d + m)
            ax[x,y].set_xlabel("cache")
            ax[x,y].set_ylabel("time")

            ax[x, y].plot( cache_per, b, label = "dgl")
            ax[x, y].plot(cache_per, [i * 3 for i in range(5)], label = "pagraph")
            ax[x, y].plot( cache_per, [i * 6 for i in range(5)], label = "gsplit")
    # handles, labels = ax[1][2].get_legend_handles_labels()

    fig.legend(ax[1, 2], labels=["dgl","pagraph", "split"],
           loc="lower center",ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom = .15)
    fig.savefig('full_figure.png', bbox_inches='tight')

if __name__=="__main__":
    dummy_plot()
