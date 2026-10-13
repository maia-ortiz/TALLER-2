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

