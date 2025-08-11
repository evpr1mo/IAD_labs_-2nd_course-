import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_blobs 
from sklearn.naive_bayes import GaussianNB, MultinomialNB 
from sklearn.metrics import confusion_matrix, precision_score, 
recall_score, f1_score, roc_curve, auc, precision_recall_curve, 
accuracy_score 
from sklearn.model_selection import GridSearchCV 
from sklearn.neural_network import MLPClassifier 

# Перший набір даних (а) 
X1, y1 = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0) 
rng = np.random.RandomState(13) 
X1_stretched = np.dot(X1, rng.randn(2, 2)) 

# Перший набір даних (а): Представлення початкових даних графічно 
plt.scatter(X1_stretched[:, 0], X1_stretched[:, 1], c=y1, cmap='viridis') 
plt.xlabel("Ознака 1") 
plt.ylabel("Ознака 2") 
plt.title("Перший набір даних (а)") 
plt.show()

# Поділ набору (а) на навчальний та валідаційний 
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_stretched, 
y1, test_size=0.2, random_state=42) 
Model1=MLPClassifier(random_state=1) 
Model1.fit(X1_train, y1_train) 

# Візуалізація границь рішень для MLPClassifier (перший набір даних) 
plot_decision_boundary(Model1, X1_stretched, y1, "MLPClassifier (Набір даних (а))")

# Прогноз 
y1_pred_M1 = Model1.predict(X1_test) 
print("\nПрогноз навчальний набору(а):", y1_pred) 

# Розрахунок додаткових результатів 
probs1_M1 = Model1.predict_proba(X1_test) 
#print("\nАпостеріорна ймовірність:", probs1_M1) 

# Розрахунок критеріїв якості 
cm1_M1, precision1_M1, recall1_M1, f1_M1, fpr1_M1, tpr1_M1, auc1_M1, 
pr1_M1 = calculate_metrics(y1_test, y1_pred_M1, probs1_M1) 
print("\nМатриця неточностей (Confusion Matrix) для MLPClassifier 
(навчальний набір):") 
print(cm1_M1) 
print(f"Точність: {precision1_M1:.2f}") 
print(f"Повнота: {recall1_M1:.2f}") 
print(f"F1-міра: {f1_M1:.2f}") 
print(f"AUC: {auc1_M1:.2f}") 

# Візуалізація ROC-кривих 
plot_roc_curve(fpr1_M1, tpr1_M1, auc1_M1, "ROC-крива для MLPClassifier (набір (а), навчальний)") 

#Візуалізація PR-кривих 
plot_precision_recall_curve(pr1_M1[0], pr1_M1[1], "PR-крива для MLPClassifier(набір (а), навчальний)") 

mlp_M1_v = MLPClassifier() 
# Навчання отриманої моделі з валідаційними параметрами 
mlp_M1_v.fit(X1_train, y1_train) 
# Оцінка моделі з валідаційними параметрами 
y1_pred_M1 = mlp_M1_v.predict(X1_test) 
 
print("\nПрогноз валідаційний набору(а):", y1_pred_M1) 
probs1_M1_v = mlp_M1_v.predict_proba(X1_test) 
 
cm1_M1_v, precision1_M1_v, recall1_M1_v, f1_M1_v, fpr1_M1_v, 
tpr1_M1_v, auc1_M1_v, pr1_M1_v = calculate_metrics( 
    y1_test, y1_pred_M1, probs1_M1_v) 
 
print("Матриця неточностей (Confusion Matrix) для MLPClassifier ((a),валідаційна):") 
print(cm1_M1_v) 
print(f"Точність: {precision1_M1_v:.2f}") 
print(f"Повнота: {recall1_M1_v:.2f}") 
print(f"F1-міра: {f1_M1_v:.2f}") 
print(f"AUC: {auc1_M1_v:.2f}") 
 
plot_roc_curve(fpr1_M1_v, tpr1_M1_v, auc1_M1_v, "ROC-крива для MLPClassifier ((a), валідаційний)") 
plot_precision_recall_curve(pr1_M1_v[0], pr1_M1_v[1], "PR-крива для MLPClassifier ((a), валідаційний)")

# Другий набір даних (б) 
np.random.seed(1) 
feature1 = np.random.randint(0, 10, size=300) 
feature2 = np.random.randint(0, 10, size=300) 
X2 = np.column_stack((feature1, feature2)) 
Y2 = np.random.randint(0, 2, size=300) 

# Другий набір даних (б): Представлення початкових даних графічно 
plt.scatter(X2[:, 0], X2[:, 1], c=Y2, cmap='viridis') 
plt.xlabel("Ознака 1") 
plt.ylabel("Ознака 2") 
plt.title("Другий набір даних (б)") 
plt.show() 

# Поділ набору даних (б) на навчальний та валідаційний 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, 
test_size=0.2, random_state=42) 

# MLPClassifier для набору даних (б) 
Model2 = MLPClassifier() 
Model2.fit(X2_train, y2_train) 

# Візуалізація границь рішень для MLPClassifier (б) 
plot_decision_boundary(Model2, X2, Y2, "MLPClassifier (Набір (б))")

#Прогноз 
y2_pred_M2 = Model2.predict(X2_test) 
print("\nПрогноз навчальний набору(б):", y2_pred_M2) 
 
# Розрахунок додаткових результатів 
probs2_M2 = Model2.predict_proba(X2_test) 
#print("\nАпостеріорна ймовірність:", probs2_M2) 
 
# Розрахунок критеріїв якості 
cm2_M2, precision2_M2, recall2_M2, f2_M2, fpr2_M2, tpr2_M2, auc2_M2, 
pr2_M2 = calculate_metrics(y2_test, y2_pred_M2, probs2_M2) 
 
print("\nМатриця неточностей (Confusion Matrix) для MLPClassifier ((б), навчальний набір):") 
print(cm2_M2) 
print(f"Точність: {precision2_M2:.2f}") 
print(f"Повнота: {recall2_M2:.2f}") 
print(f"F1-міра: {f2_M2:.2f}") 
print(f"AUC: {auc2_M2:.2f}") 
 
# Візуалізація ROC-кривих 
plot_roc_curve(fpr2_M2, tpr2_M2, auc2_M2, "ROC-крива для MLPClassifier ((б),навчальний набір)") 
 
# Візуалізація PR-кривих 
plot_precision_recall_curve(pr2[0], pr2[1], "PR-крива для MLPClassifier ((б),навчальний набір)") 

mlp_M2_v = MLPClassifier() 

# Навчання отриманої моделі з валідаційними параметрами 
mlp_M2_v.fit(X2_train, y2_train) 

# Оцінка моделі з валідаційними параметрами 
y2_pred_M2 = mlp_M2_v.predict(X2_test) 
print("\nПрогноз валідаційний набору(б):", y2_pred_M2) 
probs2_M2_v = mlp_M2_v.predict_proba(X2_test) 
cm2_M2_v, precision2_M2_v, recall2_M2_v, f2_M2_v, fpr2_M2_v, 
tpr2_M2_v, auc2_M2_v, pr2_M2_v = calculate_metrics( 
y2_test, y2_pred_M2, probs2_M2_v) 
print("Матриця неточностей (Confusion Matrix) для MLPClassifier ((б),валідаційна):") 
print(cm2_M2_v) 
print(f"Точність: {precision2_M2_v:.2f}") 
print(f"Повнота: {recall2_M2_v:.2f}") 
print(f"F1-міра: {f2_M2_v:.2f}") 
print(f"AUC: {auc2_M2_v:.2f}") 
plot_roc_curve(fpr2_M2_v, tpr2_M2_v, auc2_M2_v, "ROC-крива для MLPClassifier ((б), валідаційний)") 
plot_precision_recall_curve(pr2_M2_v[0], pr2_M2_v[1], "PR-крива для MLPClassifier ((б), валідаційний)")

# Пошук оптимальної кількості нейронів для першого набору даних 
neuron_counts = range(1, 11) 
accuracy_scores1 = [] 
for n in neuron_counts: 
mlp = MLPClassifier(hidden_layer_sizes=(n,), max_iter=1000, 
random_state=1) 
mlp.fit(X1_train, y1_train) 
y_pred = mlp.predict(X1_test) 
accuracy = accuracy_score(y1_test, y_pred) 
accuracy_scores1.append(accuracy) 

# Графік для оптимальної кількості нейронів (перший набір даних) 
plt.figure() 
plt.plot(neuron_counts, accuracy_scores1, marker='o') 
plt.xlabel("Кількість нейронів") 
plt.ylabel("Точність") 
plt.title("Оптимальна кількість нейронів (набір даних (а))") 
plt.show()

# Пошук оптимальної кількості нейронів для другого набору даних 
accuracy_scores2 = [] 
for n in neuron_counts: 
mlp = MLPClassifier(hidden_layer_sizes=(n,), max_iter=1000, 
random_state=1) 
mlp.fit(X2_train, y2_train) 
y_pred = mlp.predict(X2_test) 
accuracy = accuracy_score(y2_test, y_pred) 
accuracy_scores2.append(accuracy) 

# Графік для оптимальної кількості нейронів (другий набір даних) 
plt.figure() 
plt.plot(neuron_counts, accuracy_scores2, marker='o') 
plt.xlabel("Кількість нейронів") 
plt.ylabel("Точність") 
plt.title("Оптимальна кількість нейронів (набір даних (б))") 
plt.show()


# Динамічне додавання нейронів до скритого шару (перший набір даних) 
desired_accuracy1 = 0.95  # Задовільна точність для першого набору даних 
hidden_neurons1 = 0 
while True: 
hidden_neurons1 += 1 
mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons1,), max_iter=1000, random_state=1) 
mlp.fit(X1_train, y1_train) 
y_pred = mlp.predict(X1_test) 
accuracy1 = accuracy_score(y1_test, y_pred) 
if accuracy1 >= desired_accuracy1: 
break 
print(f"Для досягнення точності {desired_accuracy1:.2f} (набір даних (а)) було додано {hidden_neurons1} нейронів в скритий шар.") 

# Динамічне додавання нейронів до скритого шару (другий набір даних) 
desired_accuracy2 = 0.6  # Задовільна точність для другого набору даних 
hidden_neurons2 = 0 
while True: 
hidden_neurons2 += 1 
mlp = MLPClassifier(hidden_layer_sizes=(hidden_neurons2,), max_iter=1000, random_state=1) 
mlp.fit(X2_train, y2_train) 
y_pred = mlp.predict(X2_test) 
accuracy2 = accuracy_score(y2_test, y_pred) 
if accuracy2 >= desired_accuracy2: 
break 
print(f"Для досягнення точності {desired_accuracy2:.2f} (набір даних (б)) було додано {hidden_neurons2} нейронів в скритий шар.")

