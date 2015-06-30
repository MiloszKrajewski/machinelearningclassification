
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
labels = iris.target_names
# Symbols to represent the points for the three classes on the graph.
gMarkers = ["+", "_", "x"]
# Colours to represent the points for the three classes on the graph
gColours = ["blue", "magenta", "cyan"]
# The index of the class in target_names
gIndices = [0, 1, 2]
# Column indices for the two features you want to plot against each other:

def all_pairs():
    for x in range(4):
        for y in range(4):
            if x != y:
                yield (x, y)

for f1, f2 in all_pairs():
    for mark, col, i, tn in zip(gMarkers, gColours, gIndices, labels):
        plt.subplot(4, 4, f1*4 + f2 + 1)
        plt.scatter(
            x=X[iris.target == i, f1],
            y=X[iris.target == i, f2],
            marker=mark, c=col,
            label=tn)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# f1, f2, f3 = 0, 1, 3
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for mark, col, i, tn in zip(gMarkers, gColours, gIndices, labels):
#     ax.scatter(
#         xs=X[iris.target == i, f1],
#         ys=X[iris.target == i, f2],
#         zs=X[iris.target == i, f3],
#         marker=mark,
#         c=col,
#         label=tn)
#
# plt.legend(loc='upper right')
# plt.show()

