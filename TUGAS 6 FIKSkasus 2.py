import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
gaya = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
percepatan = np.array([3, 6, 9, 12, 15])

# Membuat model regresi linier
model = LinearRegression()
model.fit(gaya, percepatan)

# Membuat prediksi percepatan berdasarkan gaya
gaya_prediksi = np.array([[12]])  # Gaya yang ingin diprediksi
prediksi_percepatan = model.predict(gaya_prediksi)

# Menampilkan hasil prediksi
print(f"Prediksi Percepatan untuk Gaya 12 N: {prediksi_percepatan[0]:.2f} m/s²")

# Visualisasi data dan model
plt.scatter(gaya, percepatan, label='Data Percobaan')
plt.plot(gaya, model.predict(gaya), color='red', label='Model Regresi Linier')
plt.xlabel('Gaya (N)')
plt.ylabel('Percepatan (m/s²)')
plt.title('Prediksi Percepatan Benda berdasarkan Gaya')
plt.legend()
plt.show()
