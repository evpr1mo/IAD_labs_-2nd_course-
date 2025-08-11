import numpy as np 
import matplotlib.pyplot as plt 

# Генеруємо тестовий набір даних 
#np.random.seed(0) 
n = 6  # Кількість товарів 
m = 30  # Кількість транзакцій 
D = [set(np.random.choice(range(1, n + 1), np.random.randint(1, 6))) for _ in 
range(m)] 

# Виведемо тестовий набір даних D 
print("Тестовий набір транзакцій D:") 
for transaction in D: 
print(transaction)

# Визначаємо параметр Suppmin (поріг мінімальної підтримки) 
Suppmin = 0.1   
 
# Крок (а): Побудувати множину одноелементних частих наборів 
L1 = [] 
for i in range(1, n + 1): 
    supp_i = sum(1 for T in D if i in T) / m 
    if supp_i >= Suppmin: 
        L1.append({i}) 
 
# Результуюча множина single_наборів 
s_frequent_itemsets = L1.copy() 
 
# Вивести s_frequent_itemsets 
print("Одноелементні набори частих товарів (single_frequent_itemsets):") 
for itemset in s_frequent_itemsets: 
    print(itemset)

# Крок (б): Пошук частих наборів для k = 2, ..., n 
for k in range(2, n + 1): 
    Lk = [] 
    for F in L1: 
        for i in range(1, n + 1): 
            if i not in F: 
                F_i = F.union({i}) 
                supp_F_i = sum(1 for T in D if F_i.issubset(T)) / m 
                if supp_F_i >= Suppmin: 
                    Lk.append(F_i) 
    if not Lk: 
        break  #Крок (в), вихід з циклу 
    s_frequent_itemsets.extend(Lk) 
    L1 = Lk 
 
 #(позбавляємось від дублікатів) 
resulting_frequent_itemsets = [] 
for itemset in s_frequent_itemsets: 
    if itemset not in resulting_frequent_itemsets: 
        resulting_frequent_itemsets.append(itemset) 

# Крок (г), виводимо resulting_frequent_itemsets 
print("\n Часті набори товарів після завершення алгоритму 
(resulting_frequent_itemsets):") 
for itemset in resulting_frequent_itemsets: 
print(itemset) 


support_values = []  # глобальні ваги альтернатив 
for supp_threshold in np.linspace(0.01, 1.0, 20):  # діапазон і крок
  
# Рахуємо кількість альтернатив, які відповідають заданому порогу підтримки 
num_satisfying_alternatives = sum(1 for F in resulting_frequent_itemsets 
if 
sum(1 for T in D if F.issubset(T)) / m >= supp_threshold) 
support_values.append(num_satisfying_alternatives) 

# Побудова графіку 
plt.figure(figsize=(10, 6)) 
plt.plot(np.linspace(0.01, 1.0, 20), support_values, marker='o', linestyle='') 
plt.axvline(x=Suppmin, color='red', linestyle='--', label=f'Suppmin = {Suppmin}') 
plt.xlabel('Збереження принципу випадання альтернативи') 
plt.ylabel('Глобальні ваги альтернатив, %') 
plt.title('Градієнтний аналіз чутливості') 
plt.legend() 
plt.grid(True) 
plt.show() 
