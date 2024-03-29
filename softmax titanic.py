import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Посилання на датасет
url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"

# Завантаження датасету
titanic_data = pd.read_csv(url)
titanic_data.drop({"Cabin", "Name", "Ticket"}, axis=1, inplace=True)
# Обробка даних
# Заповнення відсутніх даних
# Заповнення колонки "Вік" середнім значенням
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
# Заповнення колонки "Порт прибуття" модою значень
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Поділ значень на ознаки та цільову змінну (тобто значення, яке й буде вираховуватися моделлю)
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

# Визначення категоріальних ознак
categorical_features = ['Sex', 'Embarked']

# one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)

# Поділ даних на тренувальну та тестувальну частини
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизація ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Визначення моделі
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Оцінка моделі (тобто точності та втрат)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy}')

