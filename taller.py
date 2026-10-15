#Grupo 2
# Estudiantes:
#- Luisa María Escobar Ayala
#- Arantxa Maía Ortiz Fernández
# Unidad 2- Taller Evaluativo: Introducción a la computación en Bioingeniería

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio 
from scipy import signal
import cv2
import nibabel as nib # Importamos esta librería para la manipulación del .nii

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

# Punto 1 - lectura y exploración de los datos clínicos
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

# Punto 2 - procesamiento y comparación de las señales ECG
def procesar_ecg(fc_csv):
    # Creamos un objeto para el paciente sano y otro para el de taquicardia
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

    # Espectro de potencia (para mostrar por qué la señal se filtra a 60 Hz)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for s, color in ((sano, "tab:blue"), (taqui, "tab:orange")):
        f1, p1 = signal.welch(s.recortar(canal, filtrada=False), s.ver_fs(), nperseg=1024)
        f2, p2 = signal.welch(s.recortar(canal, filtrada=True), s.ver_fs(), nperseg=1024)
        ax.semilogy(f1, p1, color=color, alpha=0.4, label=f'{s.label} original')
        ax.semiology(f2, p2, color=color, label=f'{s.label} filtrada')
    ax.axvline(60, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad de potencia")
    ax.set_title("Espectro antes y después del filtro Notch (línea punteada: 60 Hz)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # Comparación estadística entre ambos pacientes
    est = {"Sano": sano.estadisticas(canal, inicio, fin), 
           "Taquicardia": taqui.estadisticas(canal, inicio, fin)}
    print("=== Estadísticas del recorte (filtrado) ===")
    print(pd.DataFrame(est).T.round(4), "\n")

    # Variable derivada (frecuencia cardiaca estimada desde el ECG)
    n_s, fc_s, rr_s = estimar_fc(sano)
    n_t, fc_t, rr_t = estimar_fc(taqui)
    print("=== Frecuencia cardiaca estimada desde el ECG ===")
    print(f'Sano:       {n_s} latidos, {fc_s:.0f} lpm, intervalo R-R medio = {rr_s:.2f} s')
    print(f'Taquicardia {n_t} latidos, {fc_t:.0f} lpm, intervalo R-R medio = {rr_t:.2f} s\n ')

    print("=== Frecuencia cardiaca promedio en el CSV ===")
    print(fc_csv.round(1), "\n")

    return est, fc_s, fc_t

# Punto 3 - manipulación de la imagen de resonancia con OpenCV
def procesar_imagen():
    # Aquí cargamos el archivo .nii (DESCOMPRIMIDO) con la librería nibabel
    volumen = nib.load(RUTA_MRI)
    vol = np.asarray(volumen.dataobj, dtype=np.float32)
    print("Forma del volumen 3D:", vol.shape)

    # Hacemos un corte axial central, rotado para verlo en su orientación habitual
    corte = np.rot90(vol[:, :, vol.shape[2] // 2])

    # Pasamos a uint8 (0-255) porque OpenCV lo exige para los filtros tipo CLAHE, Canny y umbralización para adaptar el tipo de dato
    img = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Sección de filtros
    # Filtrado gaussiano (para reducir el ruido antes de segmentar)
    blurred = cv2.GaussianBlur(img, (7, 7), 1.5)

    # CLAHE (hace la ecualización adaptativa del histograma, la idea es diferenciar materia gris, blanca y LCR)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhaced = clahe.apply(img)

    # Umbralización adaptativa (para separar el tejido y fondo)
    threshold = cv2.adaptativeThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Canny (para delinear el cráneo, los ventrículos y los surcos)
    edges = cv2.Canny(img, 50, 120)

    # Apertura morfológica (para eliminar puntos de ruido y conservar estructuras grandes)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Transformación geométrica (rotamos 90 grados)
    filas, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, filas / 2), 90, 1)
    rotated = cv2.warpAffine(img, M, (cols, filas))

    # Cierre morfológico (para cerrar huecos y obtener contornos continuos)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Graficado de imagenes
    resultados =[
        ("Corte Axial original", img),
        ("Gaussiano", blurred),
        ("CLAHE", enhaced),
        ("Umbral adaptativo", threshold),
        ("Bordes Canny", edges),
        ("Apertura morfológica", opened),
        ("Rotación 90 grados", rotated),
        ("Cierre sobre bordes", closed),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, (titulo, imagen) in zip(axs.ravel(), resultados):
        ax.imshow(imagen, cmap="gray")
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    plt.subtitle("Transformaciones aplicadas a un corte de resonancia magnética", fontsize=14)
    plt.tight_layout()
    plt.show()

# Punto 4 - resumen numérico para las conclusiones
def mostrar_resumen(est, fc_s, fc_t, fc_csv, corr):
    print("=" * 70)
    print("RESUMEN PARA LAS CONCLUSIONES")
    print("=" * 70)
    print(f"ECG sano:        media = {est['sano']['media']:.3f} mV, "
          f"desv. = {est['sano']['desv_std']:.3f} mV, FC = {fc_s:.0f} lpm")
    print(f"ECG taquicardia: media = {est['taquicardia']['media']:.3f} mV, "
          f"desv. = {est['taquicardia']['desv_std']:.3f} mV, FC = {fc_t:.0f} lpm")
    print(f"FC promedio en CSV -> Sano: {fc_csv['Sano']:.0f} lpm | "
          f"Taquicardia: {fc_csv['Taquicardia']:.0f} lpm")
    print(f"Correlación FC-PAM: {corr.loc['Frecuencia_Cardiaca', 'PAM']:.2f} | "
          f"FC-Glucosa: {corr.loc['Frecuencia_Cardiaca', 'Glucosa']:.2f}")

def main():
    df, corr, fc_csv = analizar_csv()
    est, fc_s, fc_t = procesar_ecg(fc_csv)
    procesar_imagen()
    mostrar_resumen(est, fc_s, fc_t, fc_csv, corr)

if __name__ == "__main__":
    main()

# CONCLUSIONES
#
# Diferencias observadas entre las señales ECG (sano vs taquicardia):
# El ECG de taquicardia tiene casi el doble de latidos en los mismos 10 s, porque el
# intervalo R-R se acorta (de ~0.7 s a ~0.4 s): ~144 lpm frente a ~78 lpm. La media y la
# desviación estándar del recorte son mayores en la taquicardia, ya que hay más complejos
# QRS (picos grandes) por unidad de tiempo. Ambas señales tenían una fuerte componente de
# 60 Hz (ruido de la red eléctrica) que dominaba el espectro; el filtro Notch la eliminó
# sin deformar los picos R.

# Encontramos posibles relaciones entre los datos clínicos y las variables derivadas,
# la FC estimada desde el ECG sigue la tendencia del CSV (70 lpm en pacientes sanos y 124
# en pacientes con taquicardia en promedio), aunque los valores no coinciden exactamente porque son
# datos simulados y el ECG es de un solo paciente. 
# 
# En el CSV la FC se correlaciona fuerte y negativamente con la presión arterial media (-0.94) y 
# positivamente con la glucosa más alta (0.92), es decir, los pacientes con taquicardia tienen la 
# presión arterial media más baja (~83 vs ~96) y glucosa más alta (~107 vs ~89). Como los grupos no se 
# solapan en FC las correlaciones reflejan sobre todo la diferencia entre diagnósticos, con respecto a 
# la edad esta se correlaciona débilmente con las demás variables (es una asociación con datos 
# simulados, no prueba causalidad).

# Con respecto a los filtros y transformaciones de imagen, el gaussiano (7x7, sigma 1.5) suaviza el ruido
# pero difumina los bordes finos, CLAHE aumenta el contraste local y hace más visibles las estructuras internas,
# la umbralización adaptativa resalta la textura local (los surcos y bordes del cerebro), la apertura con kernel
# 2x2 solo elimina algunos puntos aislados, Canny delinea el cráneo y los surcos, y el cierre engrosa y conecta los bordes
# cercanos, por último, la rotación a 90 grados cambia la orientación de la imagen (una aplicación útil si se desea hacer
# estudios), pero recorta los bordes porque el lienzo no es cuadrado. 