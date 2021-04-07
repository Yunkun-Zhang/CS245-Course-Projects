import matplotlib.pyplot as plt
import numpy as np
import os
from brokenaxes import brokenaxes
from matplotlib.pyplot import MultipleLocator

markers = ['.', ',', 'o', '*', '_', '1', 'v', '^', '<', '>']


def read_data_ga(path, population):
    svm_final = []
    all_final = []
    with open(path, 'r') as f:
        
        lines = f.readlines()
    for i in range(100):
        svm_one_iter = []
        all_one_iter = []
        for j in range(population):
            line = lines[population*i+j].strip().split("\t")
            svm_one_iter.append(eval(line[1]))
            all_one_iter.append(eval(line[2]))
        svm_final.append(max(svm_one_iter))
        all_final.append(max(all_one_iter))
    return svm_final, all_final


def read_data_ffs(path):
    dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(8):
        line = lines[10*i]
        key = line[line.find("kernel"): line.find(")")]
        dict[key] = []
        for r in range(10):
            line = lines[10*i + r]
            dict[key].append(eval(line[11: line.find("(")]))
    return dict


def plot_figure_ga(root_path):
    plt.figure(figsize=(14, 9))
    plt.title("Best Chromosome Score during iterations", fontdict={'size': 20})
    plt.ylabel("fitness score")
    plt.xlabel("iteration")
    index = 0
    for file in os.listdir(root_path):
        if os.path.isfile(os.path.join(root_path, file)):
            path = os.path.join(root_path, file)
            info = file.split("-")
            C = info[0].split("_")[1]
            IR = info[1].split("_")[1]
            pop = info[2].split("_")[1]
            svm_score, all_score = read_data_ga(path, int(pop))
            label = f"C = {C}, IR={IR}, pop={pop}"
            x = np.arange(len(svm_score)) + 1
            plt.plot(x, svm_score, label=label, marker=markers[index % len(markers)])
            index += 1
    plt.legend(prop={'size': 15})
    plt.savefig("../report/figures/selection/ga_result.jpg")
    plt.show()
    plt.close()


def plot_figure_ffs(path):
    plt.figure(figsize=(12, 8))
    plt.title("top-k FFS Score w.r.t k", fontdict={'size': 20})
    plt.ylabel("SVM score")
    plt.xlabel("Dimensions selected")
    data = read_data_ffs(path)
    index = 0
    x = np.arange(100, 1100, 100)
    x_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for key, items in data.items():
        plt.plot(x, items, label=key, marker=markers[index % len(markers)])
        index += 1
    plt.legend(loc=(1/12, 1/8), prop={'size': 15})
    plt.savefig("../report/figures/selection/ffs_result.jpg")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    plot_figure_ga("data/ga")
