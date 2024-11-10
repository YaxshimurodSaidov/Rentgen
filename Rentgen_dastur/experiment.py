import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# 1. Ma'lumotlarni yuklash va kolonka nomlarini ajratish (delimiter=',')
data = pd.read_csv(r'C:\Users\User\Desktop\Universitet 2_kurs\suniy intellekt asoslari\dars bular\data.csv')  

# 2. Kolonkalarning nomlarini to'g'rilash
data.columns = ['ot_kuchi', 'kalibri', 'ogirligi', 'kengligi', 'uzunligi', 'ekipaj']

# 3. NaN qiymatlarni tekshirish va olib tashlash
data = data.dropna()  # Removes any rows with NaN values

# 4. X va y o'zgaruvchilarni ajratish
X = data[['kalibri', 'ogirligi', 'kengligi', 'uzunligi']].astype(float)  # Convert to float
y = data['ot_kuchi'].astype(float)  # Convert target to float

# 5. Ma'lumotlarni trening va test to'plamiga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modelni qurish va o'rgatish
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Modelni sinash
y_pred = model.predict(X_test)

# 8. Natijalarni baholash
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 9. Natijalarni chop qilish
print(f"R^2 aniqlik ko'rsatkichi: {r2:.4f}")
print(f"O'rtacha kvadrat xato (MSE): {mse:.4f}")
print(f"Modelning regressiya koeffitsientlari: {model.coef_}")
print(f"Modelning intercepti: {model.intercept_}")

# 10. Vizuallashtirish (3D scatter plot)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# For 3D scatter plot, choose three features from X
ax.scatter(X_test['kalibri'], X_test['ogirligi'], y_test, color='blue', label='Haqiqiy qiymatlar')
ax.scatter(X_test['kalibri'], X_test['ogirligi'], y_pred, color='red', label='Model taxminlari')

ax.set_xlabel('Kalibri')
ax.set_ylabel('Ogirligi')
ax.set_zlabel('Ot kuchi')

plt.title('Chiziqli regressiya natijalari (haqiqiy va taxminiy qiymatlar)')
plt.legend()
plt.show()
