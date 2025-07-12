import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
data = np.random.rand(100, 2) * 10
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, n_clusters, 1.5, error=0.005, maxiter=100, init=None)
cluster_membership = np.argmax(u, axis=0)
colors = ['blue', 'green', 'orange']
for i in range(n_clusters):
     cluster_data = data[cluster_membership == i]
     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i], label=f'Cluster {i+1}')


print("Центри кластерів:\n", cntr)
print(data)
plt.scatter(cntr[:, 0], cntr[:, 1], color='red', marker='X', s=100,
label='Cluster Centers')
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.title("Кластери")
plt.show()
plt.plot(jm, label='Objective Function')
plt.xlabel("Ітерація")
plt.ylabel("Значення цільової функції")
plt.title("Зміна значень цільової функції на кожній ітерації")
plt.legend()
plt.show()