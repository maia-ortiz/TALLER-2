#Grupo 2
# Estudiantes:
#- Luisa María Escobar Ayala
#- Arantxa Maía Ortiz Fernández
# Unidad 2- Taller Evaluativo: Introducción a la computación en Bioingeniería

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
from scipy import signal
import cv2
import nibabel as nib

# Creamos rutas de los archivos
RUTA_CSV = "clinical_data.csv"
RUTA_ECG_SANO = "ecg_sano.mat"
RUTA_ECG_TAQUI = "ecg_taqui.mat"
RUTA_MRI = "imagen_resonancia.nii" # El archivo DEBE estar descomprimido para funcionar

# Creamos la clase signal para encapsular una señal ECG
class Signal:
    def __init__(self, data, label, fs=500):
        data = np.asarray(data, dtype=float)

        # En caso de que la señal llegue como vector, se convierte a la forma (1, N)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self.__data = data
        self.label = label
        self.__fs = fs
        self.__filtrada = None

# Hacemos solo getters porque la señal original no debe ser modificada
    def ver_data(self):
        return self.__data

    def ver_fs(self):
        return self.__fs

# Filtramos Notch (rechazar banda) en 60 Hz para quitar el ruido de la red eléctrica
    def filtrar_notch(self, f0=60.0, Q=30.0):
        b, a = signal.iirnotch(w0=f0, Q=Q, fs=self.__fs)
        # Usamos filtfilt para filtrar de ida y vuelta y no desfasar la señal
        self.__filtrada = signal.filtfilt(b, a, self.__data, axis=1)
        return self.__filtrada

# Recortamos un canal y un rango de muestras (usa la señal filtrada si ya existe)
    def recortar(self, canal=0, inicio=0, fin=None, filtrada=True):
        if filtrada and self.__filtrada is not None:
            fuente = self.__filtrada
        else:
            fuente = self.__data
        return fuente[canal, inicio:fin]

# Creamos una función de media y desviación estándar del recorte
    def estadisticas(self, canal=0, inicio=0, fin=None):
        seg = self.recortar(canal, inicio, fin)
        return {"media": float(np.mean(seg)), "desv_std": float(np.std(seg))}

# Graficamos el segmento original vs el segmento filtrado
    def graficar(self, canal=0, inicio=0, fin=None, ax=None):
        if fin is None:
            fin = self.__data.shape[1]
        t = np.arange(inicio, fin) / self.__fs

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))

        ax.plot(t, self.recortar(canal, inicio, fin, filtrada=False), 
                color="gray", alpha=0.6, label="Original (con ruido de 60 Hz)")

        ax.set_title(f'ECG - Paciente {self.label}')
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Aplitud (mV)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        return ax

# Creamos una función para cargar archivos .mat y crear un objeto signal (la variable se llama "ecg")
def cargar_mat(ruta, label):
    mat = sio.loadmat(ruta)
    fs = int(mat["fs"].squeeze())
    return Signal(mat["ecg"], label, fs)

# Creamos una función para estimar la frecuencia cardiaca de los picos R
def estimar_fc(sig, canal=0):
    x = sig.recortar(canal, filtrada=True)
    picos, _ = signal.find_peaks(x, height=0.6 * np.max(x), distance=int(0.25 * sig.ver_fs()))
    duracion = x.size / sig.ver_fs()
    intervalo_rr = float(np.mean(np.diff(picos) / sig.ver_fs()))
    return len(picos), len(picos) * 60 / duracion, intervalo_rr

# Punto 1: lectura y exploración de los datos clínicos
def analizar_csv():
    df = pd.read_csv(RUTA_CSV)

    print("=== Primeras filas ===")
    print(df.head(), "\n")
    print("=== Dimensiones (filas, columnas) ===")
    print(df.shape, "\n")
    print("=== Tipos de datos ===")
    print(df.dtypes, "\n")
    print("=== Valores faltantes por columna ===")
    print(df.isna().sum(), "\n")

    # Variables numéricas (en el CSV las columnas llevan guion bajo)
    cols_num = ["Edad", "Frecuencia_Cardiaca", "PAM", "Glucosa"]

    print("=== Estadísticas descriptivas ===")
    desc = df[cols_num].agg(["mean", "std", "min", "max"]).T
    desc.columns = ["Media", "Desv. estándar", "Mínimo", "Máximo"]
    print(desc.round(2), "\n")

    # Promedios por diagnóstico, para relacionarlos luego con los ECG
    print("=== Promedio de variables por diagnóstico ===")
    por_diag = df.groupby("Diagnóstico")[cols_num].mean().round(2)
    print(por_diag, "\n")

    print("=== Pacientes por diagnóstico y género ===")
    print(pd.crosstab(df["Diagnóstico"], df["Género"]), "\n")

    print("=== Correlación entre variables numéricas ===")
    corr = df[cols_num].corr().round(2)
    print(corr, "\n")

    # Gráficas exploratorias
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # Histograma de edad
    axs[0].hist(df["Edad"], bins=12, color="steelblue", edgecolor="white")
    axs[0].set_title("Distribución de la edad")
    axs[0].set_xlabel("Edad (años)")
    axs[0].set_ylabel("Número de pacientes")

    # Boxplot de frecuencia cardiaca por diagnóstico
    datos_box = []
    for d in por_diag.index:
        datos_box.append(df.loc[df["Diagnóstico"] == d, "Frecuencia_Cardiaca"])
    axs[1].boxplot(datos_box)
    axs[1].set_xticklabels(list(por_diag.index))
    axs[1].set_title("Frecuencia cardiaca por diagnóstico")
    axs[1].set_ylabel("Frecuencia cardiaca (lpm)")

    # Mapa de calor de correlaciones
    im = axs[2].imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    axs[2].set_xticks(range(len(cols_num)))
    axs[2].set_yticks(range(len(cols_num)))
    axs[2].set_xticklabels(cols_num, rotation=45, ha="right")
    axs[2].set_yticklabels(cols_num)
    for i in range(len(cols_num)):
        for j in range(len(cols_num)):
            axs[2].text(j, i, corr.values[i, j], ha="center", va="center", fontsize=9)
    axs[2].set_title("Correlación entre variables")
    fig.colorbar(im, ax=axs[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

    fc_csv = df.groupby("Diagnóstico")["Frecuencia_Cardiaca"].mean()
    return df, corr, fc_csv

# Punto 2: procesamiento y comparación de las señales ECG
def procesar_ecg(fc_csv):
    # Instanciamos un objeto para el paciente sano y otro para el de taquicardia
    sano = cargar_mat(RUTA_ECG_SANO, "sano")
    taqui = cargar_mat(RUTA_ECG_TAQUI, "taquicardia")
    sano.filtrar_notch()
    taqui.filtrar_notch()

    # Recorte: canal 0, primeros 3 segundos (1500 muestras a 500 Hz)
    canal, inicio, fin = 0, 0, 1500

    # Segmentos de la señal procesada
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sano.graficar(canal, inicio, fin, ax=axs[0])
    taqui.graficar(canal, inicio, fin, ax=axs[1])
    plt.tight_layout()
    plt.show()