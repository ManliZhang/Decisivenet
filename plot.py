import matplotlib.pyplot as plt
import numpy as np

results = []
for line in open("results_target.txt","r"):
    eval(line)
results_target = results

x_targets = []
y_targets = []
for line in results_target:
    x_targets.append(line["target"])
    y_targets.append(line["perfs_ft"][-1])
indices = np.argsort(x_targets)
x_targets = np.take_along_axis(np.asarray(x_targets), indices, axis=0)
y_targets = np.take_along_axis(np.asarray(y_targets), indices, axis=0)

results = []
for line in open("results.txt","r"):
    eval(line)

prunes = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
prunes.reverse()
prunes = 1 - np.asarray(prunes)

plt.plot(x_targets, y_targets, label = "Dynamic width")
for i in range(len(results)):
    y = (results[i]["perfs_ft"])
    y.reverse()
    if results[i]["a"] == -1:
        label = "wd = " + str(results[i]["wd"]) + " baseline"
    else:
        label = "wd = " + str(results[i]["wd"]) + ", a = " + str(results[i]["a"]) + ", width = " + str(results[i]["width"])
    plt.plot(prunes, y, label = label)
    plt.xlabel("Kept ratio")
    plt.ylabel("Accuracy")
    plt.xscale("log")
plt.legend(bbox_to_anchor=(1.05, 1))
#plt.tight_layout()
    #plt.legend()

plt.show()
