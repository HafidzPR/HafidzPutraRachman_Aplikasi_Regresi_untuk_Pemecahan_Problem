# APLIKASI REGRESI UNTUK PEMECAHAN PROBLEM
#=======================================================

# HAFIDZ PUTRA RACHMAN ~ (21120120140096)
# METODE NUMERIK (A)
#=======================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

#=======================================================

# Muat dataset yang telah didwonload dari kaggle
file_path = r"C:\Users\hafid\OneDrive\Documents\Shadow+Essentials\UNIVERSITY-UNDIP\Programming\Aplikasi_Regresi_untuk_Pemecahan_Problem"
data = pd.read_csv(file_path)

# Periksa struktur dataset menggunakan pandas
print(data.head())

# Ekstrak data dari kolom yang relevan
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

#=======================================================

# Definisi Model Regresi linear untuk TB -> NT
linear_model_tb_nt = LinearRegression()
linear_model_tb_nt.fit(TB, NT)
NT_linear_tb_pred = linear_model_tb_nt.predict(TB)

# Plot Regresi Linier
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, color='blue', label='Data Points')
plt.plot(TB, NT_linear_tb_pred, color='red', label='Linear Regression')
plt.xlabel('Hours Studied (TB)')
plt.ylabel('Performance Index (NT)')
plt.title('Linear Regression (TB -> NT)')
plt.legend()

# Definisi Model Regresi Eksponensial untuk TB -> NT
def exponential_model(x, a, b):
    return a * np.exp(b * x)

params_tb, covariance_tb = curve_fit(exponential_model, TB.ravel(), NT)
NT_exp_tb_pred = exponential_model(TB.ravel(), *params_tb)

# Plot Regresi Eksponensial
plt.subplot(1, 2, 2)
plt.scatter(TB, NT, color='blue', label='Data Points')
plt.plot(TB, NT_exp_tb_pred, color='green', label='Exponential Regression')
plt.xlabel('Hours Studied (TB)')
plt.ylabel('Performance Index (NT)')
plt.title('Exponential Regression (TB -> NT)')
plt.legend()

#=======================================================

# Menampilkan Gambar Grafik + Data NAMA/NIM
plt.figure(figsize=(15, 5), num='APLIKASI REGRESI UNTUK PEMECAHAN PROBLEM     ----------->     HAFIDZ PUTRA RACHMAN ~ (21120120140096) ~ METODE NUMERIK (A)')
plt.figtext(0.5, 0.88, '[HAFIDZ PUTRA RACHMAN]\n[21120120140096]', ha='center', va='center', fontsize=10, color='purple')
plt.show()

#=======================================================

# Menghitung kesalahan RMS
rms_linear_tb = np.sqrt(mean_squared_error(NT, NT_linear_tb_pred))
print(f'RMS error for Linear Regression (TB -> NT): {rms_linear_tb}')
rms_exp_tb = np.sqrt(mean_squared_error(NT, NT_exp_tb_pred))
print(f'RMS error for Exponential Regression (TB -> NT): {rms_exp_tb}')

# Menguji model linear dan eksponensial
def test_models(TB_test):
    TB_test = np.array(TB_test).reshape(-1, 1)
    NT_linear_test_pred = linear_model_tb_nt.predict(TB_test)
    NT_exp_test_pred = exponential_model(TB_test.ravel(), *params_tb)
    return NT_linear_test_pred, NT_exp_test_pred

# Contoh tes linear dan eksponensial
TB_test_example = [1, 2, 3, 4, 5]
NT_linear_test_pred, NT_exp_test_pred = test_models(TB_test_example)
for i, TB_val in enumerate(TB_test_example):
    print(f'TB = {TB_val}: Linear Predicted NT = {NT_linear_test_pred[i]}, Exponential Predicted NT = {NT_exp_test_pred[i]}')