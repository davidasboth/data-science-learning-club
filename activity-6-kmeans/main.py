import pandas as pd
from matplotlib import pyplot as plt

# import custom library
from KMeansClusterer import KMeansClusterer

# load the Iris dataset
df = pd.read_csv('data/iris.csv')
df.drop('class', axis=1, inplace=True)

# plot the raw data
plt.title('Raw data')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df['petallength'], df['petalwidth'], marker='x')

# setup training variables

n_clusters = 3
iterations = 30

##################################
#           Training
##################################

kmeans = KMeansClusterer()
kmeans.train(df, k=n_clusters, n_iterations=iterations)

##################################
#         Plot clusters
##################################

# plot points coloured by their clusters
# and mark cluster centroids
fig, ax = plt.subplots()
colours = ['b', 'r', 'm', 'k', 'g']

def get_colour(i):
    return colours[i % len(colours)]

for i in range(len(kmeans.clusters)):
    ax.scatter([x[2] for x in kmeans.clusters[i]], [y[3] for y in kmeans.clusters[i]], c=get_colour(i), s=25, marker='x')

for i in range(len(kmeans.centroids)):
    ax.scatter(kmeans.centroids[i][2], kmeans.centroids[i][3], c=get_colour(i), marker='o', s=60, linewidth=0)

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Clusters (k=%d) after %d iterations' % (n_clusters, iterations))

plt.show()