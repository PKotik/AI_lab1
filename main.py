from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print("признаки:", iris.feature_names)
print("классы:", iris.target_names)
print("размерность данных:", iris.data.shape)
print("размерность меток:", iris.target.shape)
print("баланс классов:", np.bincount(iris.target))
print("\nпервые 5 строк признаков:\n", iris.data[:5])
print("первые 5 меток:", iris.target[:5])

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.3,
    random_state=109
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("баланс классов в train:", np.bincount(y_train))
print("баланс классов в test:", np.bincount(y_test))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("модель обучена.")

y_pred = knn.predict(X_test)
print("предсказания:", y_pred)
print("истинные метки:", y_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)

print("матрица ошибок:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nотчёт по классам:\n", metrics.classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# нормализация признаков
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])
pipe.fit(X_train, y_train)
y_pred_scaled = pipe.predict(X_test)

acc_scaled = metrics.accuracy_score(y_test, y_pred_scaled)
print("accuracy с нормализацией:", acc_scaled)

print("матрица ошибок:\n", metrics.confusion_matrix(y_test, y_pred_scaled))
print("\nотчёт по классам:\n", metrics.classification_report(
    y_test, y_pred_scaled, target_names=iris.target_names
))

# подбор оптимального k
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = metrics.accuracy_score(y_test, m.predict(X_test))
    accuracies.append(acc)
    print(f"k={k}: accuracy={acc:.4f}")

best_k = k_values[int(np.argmax(accuracies))]
print(f"\nлучшее k: {best_k}, accuracy: {max(accuracies):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k (число соседей)')
plt.ylabel('accuracy')
plt.title('зависимость Accuracy от k')
plt.grid(alpha=0.3)
plt.show()


# уменьшение размерности
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("доля объяснённой дисперсии по компонентам:")
print(pca.explained_variance_ratio_)
print(f"суммарно сохранено информации: {sum(pca.explained_variance_ratio_):.4f}")
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b']
for i, name in enumerate(iris.target_names):
    mask = iris.target == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=name, s=60, alpha=0.7)

plt.xlabel('главная компонента 1', fontsize=14)
plt.ylabel('главная компонента 2', fontsize=14)
plt.title('PCA набора Iris', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()