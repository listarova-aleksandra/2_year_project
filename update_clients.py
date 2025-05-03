import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

df = pd.read_csv("clients.csv")
df.head()

df.drop(columns=["id"], inplace=True)

df.shape

df.columns = df.columns.str.lower()

df.info()

df.describe()

"""##Предобработка данных

###Работа с выбросами

**Есть явные выбросы в возрасте пассажиров**
"""

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['age'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Возраст')
plt.title('Распределение возраста пассажиров')
plt.show()

df[df['age'] > 105]

df = df[(df['age'] > 0)&(df['age'] <= 105)]

"""**Сейчас возраст пассажиров адекватный**"""

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['age'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Возраст')
plt.title('Распределение возраста пассажиров')
plt.show()

"""**Удаление пропусков в target переменной**"""

df = df[df['satisfaction'] != '-']
df

df['satisfaction'].value_counts()

"""**Удаление выбросов в признаке flight distance**"""

plt.figure(figsize=(10, 5))
plt.hist(df['flight distance'], bins=200, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Дистанция полета')
plt.ylabel('Количество пассажиров')
plt.title('Распределение дистанции полета среди пассажиров')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['flight distance'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Дистанция полета')
plt.title('Распределение дистанции полета среди пассажиров')
plt.show()

outliers = df[(df["flight distance"] < 5) | (df["flight distance"] > 7000)]
outliers.shape

df = df[(df["flight distance"] >= 5) & (df["flight distance"] <= 7000)]

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['flight distance'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Дистанция полета')
plt.title('Распределение дистанции полета среди пассажиров')
plt.show()

"""**Работа с выбросами в признаках arrival departure delay in minutes ;delay in minutes**"""

plt.figure(figsize=(10, 5))
plt.hist(df['arrival delay in minutes'], bins=100, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Дистанция полета')
plt.ylabel('Количество пассажиров')
plt.title('Распределение дистанции полета среди пассажиров')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['arrival delay in minutes'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Задержка прибытия')
plt.title('Распределение задержки прибытия')
plt.show()

df = df[df['arrival delay in minutes'] < 1000]
df = df[df['departure delay in minutes'] < 1000]

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['arrival delay in minutes'], alpha=0.5, color='purple', s=10)
plt.xlabel('Пассажиры')
plt.ylabel('Задержка прибытия')
plt.title('Распределение задержки прибытия')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(df['arrival delay in minutes'], bins=100, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Дистанция полета')
plt.ylabel('Количество пассажиров')
plt.title('Распределение дистанции полета среди пассажиров')
plt.show()

"""**Чистка категориальных признаков**"""

pd.set_option('display.max_columns', None)

df[df['departure/arrival time convenient'] > 5]

exclude_columns = ['gender', 'age', 'customer type', 'type of travel', 'class', 'flight distance', 'departure delay in minutes', 'arrival delay in minutes', 'satisfaction']
service_columns = [column for column in df.columns if column not in exclude_columns]
df = df[(df[service_columns] >= 1).all(axis = 1) & (df[service_columns] <= 5).all(axis = 1)]
df

df.info()

df.describe()

"""**Обработка пропусков в признаке gender**"""

df['gender'].isna().sum()

"""**Это менее 1% данных, поэтому можно будет удалить сттроки с этими пропусками.**"""

df = df.dropna(subset = ['gender', 'customer type', 'type of travel', 'class'])

df.info()

"""###Преобразование категориальных перменных"""

label_columns = ['gender', 'customer type', 'class', 'type of travel', 'satisfaction']
label_enc = LabelEncoder()

for column in label_columns:
    df[column] = label_enc.fit_transform(df[column])

"""###Дисбаланс классов"""

df['satisfaction'].value_counts()

df_major = df[df['satisfaction'] == 0]
df_minor = df[df['satisfaction'] == 1]

df_major_downsample = resample(df_major, replace=True, n_samples=len(df_minor), random_state=22)
df = pd.concat([df_major_downsample, df_minor])

df['satisfaction'].value_counts()

df.info()

"""###Масшатибирование числовых признаков"""

df_scaled = df.copy()
scaled_exclude_columns = ['gender', 'customer type', 'type of travel', 'class', 'satisfaction']
scaled_numeric_columns = [column for column in df.columns if column not in scaled_exclude_columns]
scaler = StandardScaler()
df_scaled[scaled_numeric_columns] = scaler.fit_transform(df_scaled[scaled_numeric_columns])

df_scaled.head(5)

"""##Разделение данных

**Для немасштабированных данных**
"""

x = df.drop(columns=['satisfaction'])
y = df['satisfaction']

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
print(f"train+validation: {x_train_val.shape[0]} строк")
print(f"test: {x_test.shape[0]} строк")

"""**Для масшатированных данных**"""

x_scaled = df_scaled.drop(columns=['satisfaction'])
y_scaled = df_scaled['satisfaction']

x_scaled_train_val, x_scaled_test, y_scaled_train_val, y_scaled_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=22)
print(f"train+validation: {x_scaled_train_val.shape[0]} строк")
print(f"test: {x_scaled_test.shape[0]} строк")

"""#Ml модели

###Random Forest

**Кросс - валидация, метрики, обучение модели**
"""

def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Оценка модели:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

kv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=22)
y_train_val_pred = cross_val_predict(random_forest_model, x_train_val, y_train_val, cv=kv)
print('Оценка модели на кросс-валидации')
metrics(y_train_val, y_train_val_pred)
random_forest_model.fit(x_train_val, y_train_val)
y_test_pred = random_forest_model.predict(x_test)
print("Финальная оценка модели на отложенной тестовой выборке")
metrics(y_test, y_test_pred)

param_grid = {
    'n_estimators': [50, 70, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

kv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
random_forest = RandomForestClassifier(random_state=22)

grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=param_grid,
    cv=kv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train_val, y_train_val)
print("Лучшие параметры:", grid_search.best_params_)

best_random_forest_model = grid_search.best_estimator_
y_train_val_pred = cross_val_predict(best_random_forest_model, x_train_val, y_train_val, cv=kv)
print("Оценка модели на кросс-валидации")
metrics(y_train_val, y_train_val_pred)

best_random_forest_model.fit(x_train_val, y_train_val)
y_test_pred = best_random_forest_model.predict(x_test)
print("Финальная оценка модели на отложенной тестовой выборке")
metrics(y_test, y_test_pred)

"""**Теперьь организую идею  с вероятностями при помощи predict_proba**"""

def st_pred(st, dst):
    if st >= dst:
        return f'Вам понравился полет с вероятностью {st * 100:.2f}%'
    else:
        return f'Вам не понравился полет с вероятностью {dst * 100:.2f}%'

y_proba = random_forest_model.predict_proba(x)

df['satisfaction with the flight'] = y_proba[:, 1]
df['dissatisfaction with the flight'] = y_proba[:, 0]

df['result'] = df.apply(
    lambda row: st_pred(row['satisfaction with the flight'], row['dissatisfaction with the flight']),
    axis=1)

joblib.dump(best_random_forest_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
