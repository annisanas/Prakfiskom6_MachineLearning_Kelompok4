import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
waktu = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
ketinggian = np.array([2, 4, 7, 11, 16, 22])

# Membuat model regresi linier
model = LinearRegression()
model.fit(waktu, ketinggian)

# Membuat prediksi ketinggian bola pada waktu 10 detik
waktu_prediksi = np.array([[10]])
prediksi_ketinggian = model.predict(waktu_prediksi)

# Menampilkan hasil prediksi
print(f"Prediksi Ketinggian Bola pada Waktu 10 detik: {prediksi_ketinggian[0]:.2f} meter")

# Menampilkan grafik data dan model regresi linier
plt.scatter(waktu, ketinggian, label='Data')
plt.plot(waktu, model.predict(waktu), color='red', label='Regresi Linier')
plt.scatter(waktu_prediksi, prediksi_ketinggian, color='green', label='Prediksi')
plt.xlabel('Waktu (detik)')
plt.ylabel('Ketinggian (meter)')
plt.title('Regresi Linier untuk Prediksi Ketinggian Bola')
plt.legend()
plt.show()
