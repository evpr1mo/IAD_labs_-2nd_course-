import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_blobs 
from sklearn.naive_bayes import GaussianNB, MultinomialNB 
from sklearn.metrics import confusion_matrix, precision_score, 
recall_score, f1_score, roc_curve, auc, precision_recall_curve 
from sklearn.model_selection import GridSearchCV 

# Перший набір даних (а) 
X1, y1 = make_blobs(n_samples=400, centers=4, cluster_std=0.60, 
random_state=0) 
rng = np.random.RandomState(13) 
X1_stretched = np.dot(X1, rng.randn(2, 2)) 

# Перший набір даних (а): Представлення початкових даних графічно 
plt.scatter(X1_stretched[:, 0], X1_stretched[:, 1], c=y1, 
cmap='viridis') 
plt.xlabel("Ознака 1") 
plt.ylabel("Ознака 2") 
plt.title("Перший набір даних (а)") 
plt.show() 

# Поділ набору даних на навчальний та валідаційний 
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_stretched, 
y1, test_size=0.2, random_state=42) 
 
# Gaussian Naive Bayes для першого набору даних (а) 
gnb1 = GaussianNB() 
gnb1.fit(X1_train, y1_train) 
 
# Функція для візуалізації границь рішень 
def plot_decision_boundary(model, X, y, title): 
    plt.figure() 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.') 
    h = .02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, 
y_max, h)) 
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8) 
    plt.title(title) 
    plt.show() 
 
# Візуалізація границь рішень для Gaussian Naive Bayes (перший набір даних) 
plot_decision_boundary(gnb1, X1_stretched, y1, "Gaussian Naive Bayes (Набір даних 1)") 
                       
# Прогноз 
y1_pred = gnb1.predict(X1_test)

# Оцінка перенавчання 
def check_overfitting(model, X_train, y_train, X_test, y_test): 
train_acc = model.score(X_train, y_train) 
test_acc = model.score(X_test, y_test) 
return train_acc, test_acc 
train_acc1, test_acc1 = check_overfitting(gnb1, X1_train, y1_train, 
X1_test, y1_test) 
print("Перенавчання для Gaussian Naive Bayes (Набір даних 1):") 
print(f"Точність на навчальних даних: {train_acc1:.2f}") 
print(f"Точність на валідаційних даних: {test_acc1:.2f}")

# Розрахунок додаткових результатів 
probs1 = gnb1.predict_proba(X1_test) 

# Розрахунок критеріїв якості 
def calculate_metrics(y_true, y_pred, probs): 
    cm = confusion_matrix(y_true, y_pred) 
    precision = precision_score(y_true, y_pred, average='weighted') 
    recall = recall_score(y_true, y_pred, average='weighted') 
    f1 = f1_score(y_true, y_pred, average='weighted') 
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1],pos_label=1) 
    auc_score = auc(fpr, tpr) 
    precision_recall = precision_recall_curve(y_true, probs[:, 1], 
pos_label=1) 
    return cm, precision, recall, f1, fpr, tpr, auc_score, 
precision_recall 
 
cm1, precision1, recall1, f1_1, fpr1, tpr1, auc1, pr1 = 
calculate_metrics(y1_test, y1_pred, probs1) 
print("\nМатриця неточностей (Confusion Matrix) для Gaussian Naive 
Bayes (навчальний набір):") 
print(cm1) 
print(f"Точність: {precision1:.2f}") 
print(f"Повнота: {recall1:.2f}") 
print(f"F1-міра: {f1_1:.2f}") 
print(f"AUC: {auc1:.2f}") 
 
# Візуалізація ROC-кривих 
def plot_roc_curve(fpr, tpr, auc, title): 
    plt.figure() 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = 
{auc:.2f}') 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('Відсоток false-позитивних') 
    plt.ylabel('Відсоток правильних') 
    plt.title(title) 
    plt.legend(loc="lower right") 
    plt.show() 

plot_roc_curve(fpr1, tpr1, auc1, "ROC-крива для Gaussian Naive Bayes 
(навчальний)") 
# Візуалізація PR-кривих 
def plot_precision_recall_curve(precision, recall, title): 
plt.figure() 
plt.plot(recall, precision, color='darkorange', lw=2) 
plt.xlabel('Повнота') 
plt.ylabel('Точність') 
plt.title(title) 
plt.show() 
plot_precision_recall_curve(pr1[0], pr1[1], "PR-крива для Gaussian 
Naive Bayes (навчальний)") 


# Решітчастий пошук для Gaussian Naive Bayes 
param_grid = {'var_smoothing': np.logspace(0, -9, num=100)} 
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5) 
grid_search.fit(X1_train, y1_train) 
best_params = grid_search.best_params_ 
best_gnb = grid_search.best_estimator_ 
print("Найкращі гіперпараметри для Gaussian Naive Bayes (Набір даних 
1):", best_params) 

# Навчання моделі з найкращими параметрами 
best_gnb.fit(X1_train, y1_train) 

# Оцінка моделі з найкращими параметрами 
y1_pred_best = best_gnb.predict(X1_test) 
print("\nПрогноз валідаційний набору(а):", y1_pred_best) 
probs1_best = best_gnb.predict_proba(X1_test) 
cm1_best, precision1_best, recall1_best, f1_1_best, fpr1_best, 
tpr1_best, auc1_best, pr1_best = calculate_metrics( 
y1_test, y1_pred_best, probs1_best) 
print("Матриця неточностей (Confusion Matrix) для Gaussian Naive Bayes 
(з найкращими параметрами):") 
print(cm1_best) 
print(f"Точність: {precision1_best:.2f}") 
print(f"Повнота: {recall1_best:.2f}") 
print(f"F1-міра: {f1_1_best:.2f}") 
print(f"AUC: {auc1_best:.2f}") 
plot_roc_curve(fpr1_best, tpr1_best, auc1_best, "ROC-крива для 
Gaussian Naive Bayes (валідаційний)") 
plot_precision_recall_curve(pr1_best[0], pr1_best[1], "PR-крива для 
Gaussian Naive Bayes (валідаційний)")


#--------------------------2й Датасет-------------------------------------# 
np.random.seed(1) 
# Генеруємо випадковий count для ознаки 1 
feature1 = np.random.randint(0, 10, size=300) 
# Генеруємо випадковий count для ознаки 2 
feature2 = np.random.randint(0, 10, size=300) 
# Побудова вибірки, де кожний рядок - це вектор ознак 
X2 = np.column_stack((feature1, feature2)) 
# Генеруємо вектор класів (0 або 1) 
Y2 = np.random.randint(0, 2, size=300) 
# Другий набір даних (б): Представлення початкових даних графічно 
plt.scatter(X2[:, 0], X2[:, 1], c=Y2, cmap='viridis') 
plt.xlabel("Ознака 1") 
plt.ylabel("Ознака 2") 
plt.title("Другий набір даних (б)") 
plt.show() 

# Поділ набору даних на навчальний та валідаційний 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, 
test_size=0.2, random_state=42) 
 
# Multinomial Naive Bayes для другого набору даних (б) 
mnb2 = MultinomialNB() 
mnb2.fit(X2_train, y2_train) 
 
# Візуалізація границь рішень для Multinomial Naive Bayes (другий 
набір даних) 
plot_decision_boundary(mnb2, X2, Y2, "Multinomial Naive Bayes (Набір 
даних б)")

#Прогноз 
y2_pred = mnb2.predict(X2_test) 
print("\nПрогноз навчальний набору(б):", y2_pred) 
 
#Оцінка перенавчання 
train_acc2, test_acc2 = check_overfitting(mnb2, X2_train, y2_train, 
X2_test, y2_test) 
 
print("\nПеренавчання для Multinomial Naive Bayes (Набір даних 2):") 
print(f"Точність на навчальних даних: {train_acc2:.2f}") 
print(f"Точність на валідаційних даних: {test_acc2:.2f}") 
# Розрахунок додаткових результатів 
probs2 = mnb2.predict_proba(X2_test) 


# Розрахунок критеріїв якості 
cm2, precision2, recall2, f2, fpr2, tpr2, auc2, pr2 = 
calculate_metrics(y2_test, y2_pred, probs2) 
print("\nМатриця неточностей (Confusion Matrix) для MultinomialNB 
(навчальний набір):") 
print(cm2) 
print(f"Точність: {precision2:.2f}") 
print(f"Повнота: {recall2:.2f}") 
print(f"F1-міра: {f2:.2f}") 
print(f"AUC: {auc2:.2f}") 

# Візуалізація ROC-кривих 
plot_roc_curve(fpr2, tpr2, auc2, "ROC-крива для Multinomial Naive 
Bayes (навчальний набір)") 

# Візуалізація PR-кривих 
plot_precision_recall_curve(pr2[0], pr2[1], "PR-крива для Multinomial 
Naive Bayes (навчальний набір)") 

# Решітчастий пошук для Multinomial Naive Bayes 
param_grid = {'alpha': np.logspace(0, 1, num=10)} 
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5) 
grid_search.fit(X2_train, y2_train) 
best_params = grid_search.best_params_ 
best_mnb2 = grid_search.best_estimator_ 
 
print("Найкращі гіперпараметри для Multinomial Naive Bayes (Набір даних (б)):", best_params)

# Навчання моделі з найкращими параметрами 
best_mnb2.fit(X2_train, y2_train) 
 
# Оцінка моделі з найкращими параметрами 
y2_pred_best = best_mnb2.predict(X2_test) 
print("\nПрогноз валідаційний набору(б):", y2_pred_best) 
 
probs2_best = best_mnb2.predict_proba(X2_test) 
 
cm2_best, precision2_best, recall2_best, f2_best, fpr2_best, 
tpr2_best, auc2_best, pr2_best = calculate_metrics( 
    y2_test, y2_pred_best, probs2_best) 
 
print("\nМатриця неточностей (Confusion Matrix) для Multinomial Naive 
Bayes (з найкращими параметрами):") 
print(cm2_best) 
print(f"Точність: {precision2_best:.2f}") 
print(f"Повнота: {recall2_best:.2f}") 
print(f"F1-міра: {f2_best:.2f}") 
print(f"AUC: {auc2_best:.2f}") 
 
# Візуалізація ROC-кривих 
plot_roc_curve(fpr2_best, tpr2_best, auc2_best, "ROC-крива для 
Multinomial Naive Bayes (навчальний набір)") 
 
# Візуалізація PR-кривих 
plot_precision_recall_curve(pr2_best[0], pr2_best[1], "PR-крива для 
Multinomial Naive Bayes (навчальний набір)")
