from sklearn import datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import adjusted_rand_score, v_measure_score 
from sklearn.metrics.pairwise import pairwise_distances 
import time 
import numpy as np 
import matplotlib.pyplot as plt

# Завдання (1а): Завантаження та візуалізація даних 
X, y = datasets.make_moons(n_samples=200, noise=0.05) 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis') 
plt.title("Початкові дані (make_moons)") 
plt.show() 

# Завдання (2а): Створення моделі AgglomerativeClustering 
model = AgglomerativeClustering(n_clusters=2) 
# Завдання (3а): Кластеризація даних 
start_time = time.time() 
y_pred = model.fit_predict(X) 
elapsed_time = time.time() - start_time 
print(f"Час кластеризації: {elapsed_time:.2f} сек")

# Завдання (4а): Візуалізація розбиття на кластери 
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis') 
plt.title("Результат кластеризації (AgglomerativeClustering)") 
plt.show() 

# Завдання (5а): Оцінка швидкості та якості для великого набору даних 
make_moons 
X_large, y_large = datasets.make_moons(n_samples=20000, noise=0.05) 
model_large = AgglomerativeClustering(n_clusters=2) 
start_time = time.time() 
y_pred_large = model_large.fit_predict(X_large) 
elapsed_time_large = time.time() - start_time 
print(f"Час кластеризації для великого набору даних: 
{elapsed_time_large:.2f} сек")

# Візуалізація результатів для model_large на наборі даних make_moons 
plt.scatter(X_large[:, 0], X_large[:, 1], c=y_pred_large, 
cmap='viridis') 
plt.title("Результат кластеризації (model_large)") 
plt.show()

def estimated_num_clusters(model, X): 
return model.n_clusters_ 
# Функція для отримання масиву кластерів 
def get_clusters(X, labels): 
unique_labels = np.unique(labels) 
clusters = [] 
for label in unique_labels: 
cluster = X[labels == label] 
clusters.append(cluster) 
return clusters 
# Функція для обчислення матриці відстаней між кластерами 
def cluster_distance_matrix(clusters): 
    n_clusters = len(clusters) 
    dist_matrix = np.zeros((n_clusters, n_clusters)) 
    for i in range(n_clusters): 
        for j in range(i+1, n_clusters): 
            dist_matrix[i, j] = pairwise_distances(clusters[i], 
clusters[j]).mean() 
            dist_matrix[j, i] = dist_matrix[i, j] 
    return dist_matrix 

# Завдання (6+7 (а)): Побудова альтернативних моделей для make_moons 
models_alternative = [] 
linkage_methods = ['ward', 'single', 'average', 'complete'] 
 
for method in linkage_methods: 
    model_alternative = AgglomerativeClustering(n_clusters=2, 
linkage=method) 
    models_alternative.append(model_alternative) 
 
# Візуалізація альтернативних моделей та метрик + матриця 
pairwise_distances для кластерів 
for i, model in enumerate(models_alternative): 
    y_pred_alternative = model.fit_predict(X) 
    clusters = get_clusters(X, y_pred_alternative) 
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_alternative, 
cmap='viridis') 
    plt.title(f"Результат кластеризації {linkage_methods[i]} 
(make_moons)") 
    plt.show() 
    estimated_clusters = estimated_num_clusters(model, X) 
    ari = adjusted_rand_score(y, y_pred_alternative) 
    v_measure = v_measure_score(y, y_pred_alternative) 
    pairwise_distances_matrix = cluster_distance_matrix(clusters) 
    print(f"\nEstimated number of clusters {linkage_methods[i]} 
(make_moons): {estimated_clusters}") 
    print(f"Adjusted Rand Index {linkage_methods[i]} (make_moons): 
{ari:.2f}") 
    print(f"V-measure {linkage_methods[i]} (make_moons): 
{v_measure:.2f}") 
    print(f"\nМатриця відстаней кластеризації {linkage_methods[i]} 
(make_moons):\n") 
    print(pairwise_distances_matrix) 


# Завдання (8а): Аналіз стабільності розбиття 
# Видалення одного об'єкта та повторна кластеризація 
idx_to_remove = np.random.randint(0, len(X)) 
X_removed = np.delete(X, idx_to_remove, axis=0) 
y_removed = np.delete(y, idx_to_remove) 
y_pred_removed = model.fit_predict(X_removed) 
clusters_r = get_clusters(X_removed, y_pred_removed) 
plt.scatter(X_removed[:, 0], X_removed[:, 1], c=y_pred_removed, 
cmap='viridis') 
plt.title("Результат кластеризації ward (make_moons) з видаленим 
об'єктом") 
plt.show() 
pairwise_distances_matrix_r = cluster_distance_matrix(clusters_r) 
estimated_clusters_r = estimated_num_clusters(model, X_removed) 
ari_r = adjusted_rand_score(y_removed, y_pred_removed) 
v_measure_r = v_measure_score(y_removed, y_pred_removed) 
print(f"\nEstimated number of clusters після видалення об'єкта 
(make_moons): {estimated_clusters_r}") 
print(f"Adjusted Rand Index після видалення об'єкта (make_moons): {ari_r:.2f}") 
print(f"V-measure після видалення об'єкта (make_moons): {v_measure_r:.2f}") 
print(f"\nМатриця відстаней кластеризації 'ward' після видалення об'єкта (make_moons):\n") 
print(pairwise_distances_matrix_r) 


# Завдання (1б): Завантаження та візуалізація даних 
iris = datasets.load_iris() 
X = iris.data 
y = iris.target 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis') 
plt.title("Початкові дані (load_iris)") 
plt.show() 

# Завдання (2б): Створення моделі AgglomerativeClustering 
model2 = AgglomerativeClustering(n_clusters=3) 

# Завдання (3б): Кластеризація даних 
start_time = time.time() 
y_pred2 = model2.fit_predict(X) 
elapsed_time = time.time() - start_time 
print(f"Час кластеризації: {elapsed_time:.2f} сек") 

# Завдання (4б): Візуалізація розбиття на кластери 
plt.scatter(X[:, 0], X[:, 1], c=y_pred2, cmap='viridis') 
plt.title("Результат кластеризації (AgglomerativeClustering)") 
plt.show()

# Завдання (5б): Оцінка швидкості та якості для великого набору даних 
X_large, y_large = datasets.make_classification(n_samples=20000, 
n_features=4, n_informative=2, n_redundant=1, n_clusters_per_class=1, 
random_state=42) 
model_large2 = AgglomerativeClustering(n_clusters=3) 
start_time = time.time() 
y_pred_large2 = model_large2.fit_predict(X_large) 
elapsed_time_large = time.time() - start_time 
print(f"Час кластеризації для великого набору даних (load_iris): 
{elapsed_time_large:.2f} сек") 
plt.scatter(X_large[:, 0], X_large[:, 1], c=y_pred_large2, 
cmap='viridis') 
plt.title("Результат кластеризації (AgglomerativeClustering) великого набору load_iris") 
plt.show() 

# Завдання (6+7 (б)): Побудова альтернативних моделей для load_iris 
models_iris_alternative = [] 
 
for method in linkage_methods: 
    model_iris_alternative = AgglomerativeClustering(n_clusters=3, 
linkage=method) 
    models_iris_alternative.append(model_iris_alternative) 
 
# Візуалізація альтернативних моделей та метрик + матриця 
pairwise_distances для кластерів 
for i, model2 in enumerate(models_iris_alternative): 
    y_pred_iris_alternative = model2.fit_predict(X) 
    clusters_iris = get_clusters(X, y_pred_iris_alternative) 
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_iris_alternative, 
cmap='viridis') 
    plt.title(f"Результат кластеризації {linkage_methods[i]} 
(load_iris)") 
    plt.show() 
    estimated_clusters_iris = estimated_num_clusters(model2, X) 
    ari_iris = adjusted_rand_score(y, y_pred_iris_alternative) 
    v_measure_iris = v_measure_score(y, y_pred_iris_alternative) 
    pairwise_distances_matrix_iris = 
cluster_distance_matrix(clusters_iris) 
    print(f"\nEstimated number of clusters {linkage_methods[i]} (load_iris): {estimated_clusters_iris}") 
    print(f"Adjusted Rand Index {linkage_methods[i]} (load_iris): {ari_iris:.2f}") 
print(f"V-measure {linkage_methods[i]} (load_iris): {v_measure_iris:.2f}") 
print(f"\nМатриця відстаней кластеризації {linkage_methods[i]} (load_iris):\n") 
print(pairwise_distances_matrix_iris) 

# Завдання (8б): Аналіз стабільності розбиття 
# Видалення одного об'єкта та повторна кластеризація 
idx_to_remove = np.random.randint(0, len(X)) 
X_removed = np.delete(X, idx_to_remove, axis=0) 
y_removed = np.delete(y, idx_to_remove) 
y_pred_removed = model2.fit_predict(X_removed) 
clusters_r = get_clusters(X_removed, y_pred_removed) 
plt.scatter(X_removed[:, 0], X_removed[:, 1], c=y_pred_removed, 
cmap='viridis') 
plt.title("Результат кластеризації ward (load_iris) з видаленим 
об'єктом") 
plt.show() 
pairwise_distances_matrix_r = cluster_distance_matrix(clusters_r) 
estimated_clusters_r = estimated_num_clusters(model2, X_removed) 
ari_r = adjusted_rand_score(y_removed, y_pred_removed) 
v_measure_r = v_measure_score(y_removed, y_pred_removed) 
print(f"\nEstimated number of clusters після видалення об'єкта (load_iris): {estimated_clusters_r}") 
print(f"Adjusted Rand Index після видалення об'єкта (load_iris): {ari_r:.2f}") 
print(f"V-measure після видалення об'єкта (load_iris): {v_measure_r:.2f}") 
print(f"\nМатриця відстаней кластеризації 'ward' після видалення об'єкта (load_iris):\n") 
print(pairwise_distances_matrix_r) 
