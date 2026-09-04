# -*- coding: utf-8 -*-
"""
PIPELINE EEG ORGANIZADO PARA TKINTER
- No ejecuta procesamiento al importar
- Mantiene filtros, constantes y cálculos del código original
- La interfaz puede llamar:
    carpeta_cache = procesar_archivo(nombre_archivo, carpeta_base=..., logger=...)
    ruta_pdf = generar_informe_desde_cache(carpeta_cache, logger=...)
"""

import os
import re
import gc
import datetime
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, spectrogram, find_peaks
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from fpdf import FPDF

from matplotlib.patches import Patch


def _trapz(y, x=None, axis=-1):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x=x, axis=axis)



# =========================
# CONFIG
# =========================
CARPETA_BASE = os.environ.get("NEUROX_DATOS_DIR", os.getcwd())
NOMBRE_DAT = "DCL 007.dat"
# NOMBRE_DAT = "DCL 100.dat"

n_canales = 64
NOMBRES_CANALES_64 = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "M1",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "M2",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

# Fallback solo para el informe si por alguna razón falta meta.txt.
# En el flujo normal, fs se toma del .dap y luego del meta.txt.
fs = 500

bandas = {
    "Delta": (1.0, 3.0),
    "Theta": (3.0, 8.0),
    "Alfa":  (8.0, 12.0),
    "Beta":  (12.0, 30.0),
}

# CONFIGURACION FILTRO
lowcut = 1
highcut = 40
order = 4
AUTO_DETECTAR_PICOS_TECNICOS = True
AUTO_PICOS_RANGO_BUSQUEDA = (4.0, 35.0)
AUTO_PICOS_EXCLUIR_FRECUENCIAS = [60.0]
AUTO_PICOS_FACTOR_ATENUACION = 0.60
AUTO_PICOS_MAX_FRECUENCIAS = 6
AUTO_PICOS_UMBRAL_PROP_CANALES = 0.40
AUTO_PICOS_FACTOR_LOCAL = 2.5
AUTO_PICOS_VENTANA_LOCAL_HZ = 2.0
AUTO_PICOS_EXCLUSION_CENTRAL_HZ = 0.5
AUTO_PICOS_ANCHO_MAX_HZ = 1.0
AUTO_PICOS_BANDA_DELTA = bandas["Delta"]
AUTO_PICOS_USAR_CRITERIO_DELTA = True
AUTO_PICOS_USAR_CRITERIO_LOCAL = True
AUTO_PICOS_CORRECCION_POR_CANAL = False
AUTO_PICOS_MAX_FRECUENCIAS_POR_CANAL = 4
AUTO_PICOS_FACTOR_LOCAL_CANAL = 3.0
AUTO_PICOS_FACTOR_DELTA_CANAL = 1.0
AUTO_PICOS_MIN_PROMINENCIA_CANAL = 3.0

# CONFIGURACION WAVELET
WAVELET_NAME = "db4"
WAVELET_NIVEL = 4

# Ventana cruda a guardar
VENTANA_INICIO_S = 60
VENTANA_DURACION_S = 5



RANGO_TOTAL_WELCH = (1, 30)

RATIOS_WELCH_PRINCIPALES = {
    "Theta_Alfa": {
        "label": "Theta/Alfa",
        "numerador": ("Theta",),
        "denominador": ("Alfa",),
    },
    "Delta_Alfa": {
        "label": "Delta/Alfa",
        "numerador": ("Delta",),
        "denominador": ("Alfa",),
    },
    "Theta_Beta": {
        "label": "Theta/Beta",
        "numerador": ("Theta",),
        "denominador": ("Beta",),
    },
    "Lentificacion": {
        "label": "Lentificación",
        "numerador": ("Delta", "Theta"),
        "denominador": ("Alfa", "Beta"),
    },
}

REGIONES_EEG_64 = {
    "Frontal": [
        "FP1", "FPZ", "FP2", "AF3", "AF4",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"
    ],
    "Frontocentral": [
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"
    ],
    "Central/Temporal": [
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"
    ],
    "Temporo-parietal": [
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"
    ],
    "Parietal": [
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"
    ],
    "Occipital": [
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2"
    ],
    "Referencia/Periféricos": [
        "M1", "M2", "CB1", "CB2"
    ]
}

# Rangos normales de referencia (% energía relativa)
rangos_referencia = {
    "Delta": (0, 25),
    "Theta": (0, 20),
    "Alfa":  (15, 35),# estaba en 20-40
    "Beta":  (10, 30),
    # "Gamma": (1, 10)
}


# =========================
# HELPERS
# =========================
def _mklogger(logger):
    if logger is None:
        return print
    return logger


def construir_rutas_trabajo(nombre_archivo, carpeta_base):
    """
    Construye todas las rutas del CACHE para un archivo dado.
    """
    nombre_limpio = os.path.splitext(os.path.basename(nombre_archivo))[0]
    nombre_limpio = nombre_limpio.replace(" ", "_").replace(".", "_")

    carpeta_cache_root = os.path.join(carpeta_base, "CACHE")
    carpeta_archivo = os.path.join(carpeta_cache_root, nombre_limpio)

    rutas = {
        "cache_root": carpeta_cache_root,
        "archivo": carpeta_archivo,
        "crudo": os.path.join(carpeta_archivo, "crudo"),
        "fft": os.path.join(carpeta_archivo, "fft"),
        "espectrograma": os.path.join(carpeta_archivo, "espectrograma"),
        "filtrado": os.path.join(carpeta_archivo, "filtrado"),
        "ica": os.path.join(carpeta_archivo, "ica"),
        "bandas": os.path.join(carpeta_archivo, "bandas"),
        "bandas_filtrado": os.path.join(carpeta_archivo, "bandas_filtrado"),
        "canales_atipicos": os.path.join(carpeta_archivo, "canales_atipicos"),
        "wavelet": os.path.join(carpeta_archivo, "wavelet"),
        "analisis_welch": os.path.join(carpeta_archivo, "analisis_welch"),
        "informe": os.path.join(carpeta_archivo, "informe"),
    }
    return rutas


def crear_carpetas_trabajo(rutas):
    """
    Crea todas las carpetas necesarias del cache.
    """
    for c in rutas.values():
        os.makedirs(c, exist_ok=True)


# =========================
# FILTROS / LECTURA
# =========================
def filtrar_banda(datos, fs, lowcut, highcut, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, datos, axis=1)


def leer_dap(ruta_dap):
    if not os.path.exists(ruta_dap):
        raise FileNotFoundError(f"No se encontró el archivo .dap: {ruta_dap}")

    with open(ruta_dap, "r", encoding="utf-8-sig", errors="ignore") as f:
        txt = f.read()

    def extraer_clave(clave, cast=str, default=None):
        m = re.search(rf"^\s*{re.escape(clave)}\s*=\s*(.+?)\s*$", txt, re.MULTILINE)
        if not m:
            return default
        valor = m.group(1).strip()
        try:
            return cast(valor)
        except Exception:
            return default

    data_format = extraer_clave("DataFormat", str, "FLOAT")
    num_channels = extraer_clave("NumChannels", int, None)
    num_samples = extraer_clave("NumSamples", int, None)
    fs_dap = extraer_clave("SampleFreqHz", float, None)
    data_unit = extraer_clave("DataUnit", str, "uV")
    samp_order = extraer_clave("DataSampOrder", str, "SAMP")
    sample_time_usec = extraer_clave("SampleTimeUsec", float, None)

    data_format_up = str(data_format).strip().upper()

    if data_format_up in ("FLOAT", "FLOAT32"):
        dtype = np.float32
    elif data_format_up in ("SHORT", "INT16", "INTEGER16"):
        dtype = np.int16
    elif data_format_up in ("DOUBLE", "FLOAT64"):
        dtype = np.float64
    else:
        raise ValueError(f"DataFormat no soportado en .dap: {data_format}")

    if fs_dap is None and sample_time_usec is not None and sample_time_usec > 0:
        fs_dap = 1e6 / sample_time_usec

    if num_channels is None:
        raise ValueError("No se pudo leer NumChannels desde el .dap")
    if num_samples is None:
        raise ValueError("No se pudo leer NumSamples desde el .dap")
    if fs_dap is None:
        raise ValueError("No se pudo leer SampleFreqHz desde el .dap")

    return {
        "dtype": dtype,
        "data_format": data_format_up,
        "num_channels_total": int(num_channels),
        "num_samples": int(num_samples),
        "fs": float(fs_dap),
        "data_unit": str(data_unit),
        "samp_order": str(samp_order).strip().upper(),
    }


def cargar_datos_eeg(ruta_dat, fs_fijo=None, n_canales_eeg=64, logger=None):
    log = _mklogger(logger)

    ruta_dap = os.path.splitext(ruta_dat)[0] + ".dap"
    meta_dap = leer_dap(ruta_dap)

    dtype = meta_dap["dtype"]
    n_canales_total = meta_dap["num_channels_total"]
    n_muestras = meta_dap["num_samples"]
    unidad = meta_dap["data_unit"]
    samp_order = meta_dap["samp_order"]

    datos = np.fromfile(ruta_dat, dtype=dtype)

    esperados = n_canales_total * n_muestras
    if datos.size != esperados:
        raise ValueError(
            f"Tamaño inesperado del .dat. "
            f"Esperado={esperados}, encontrado={datos.size}. "
            f"dtype={dtype}, canales={n_canales_total}, muestras={n_muestras}"
        )

    if samp_order == "SAMP":
        datos = datos.reshape(n_muestras, n_canales_total).T
    else:
        datos = datos.reshape(n_canales_total, n_muestras)

    if n_canales_eeg > n_canales_total:
        raise ValueError(
            f"Se pidieron {n_canales_eeg} canales EEG, "
            f"pero el archivo solo tiene {n_canales_total}"
        )

    datos_eeg = datos[:n_canales_eeg].astype(np.float32, copy=False)

    fs_real = float(fs_fijo) if fs_fijo is not None else float(meta_dap["fs"])
    tiempo = np.arange(datos_eeg.shape[1], dtype=np.float32) / np.float32(fs_real)
    duracion = datos_eeg.shape[1] / fs_real

    log(f"Archivo .dat: {ruta_dat}")
    log(f"Archivo .dap: {ruta_dap}")
    log(f"Detectado dtype desde .dap: {dtype}")
    log(f"DataFormat: {meta_dap['data_format']}")
    log(f"Canales totales en archivo: {n_canales_total}")
    log(f"Canales EEG usados: {n_canales_eeg}")
    log(f"Muestras por canal: {datos_eeg.shape[1]}")
    log(f"Frecuencia de muestreo usada: {fs_real:.1f} Hz")
    log(f"Duración total: {duracion:.2f} s")
    log(f"Unidad reportada por .dap: {unidad}")
    log(f"Formato final: {datos_eeg.shape}")

    return datos_eeg, tiempo, fs_real, meta_dap


def guardar_crudo_y_tiempo(datos_eeg, tiempo, fs_real, carpeta_crudo, logger=None):
    """
    Guarda la señal cruda completa y el vector de tiempo.
    """
    log = _mklogger(logger)

    duracion_total = datos_eeg.shape[1] / fs_real
    log(f"Duración total de la muestra: {duracion_total:.2f} s")

    os.makedirs(carpeta_crudo, exist_ok=True)

    np.save(os.path.join(carpeta_crudo, "eeg_crudo.npy"), datos_eeg.astype(np.float32, copy=False))
    np.save(os.path.join(carpeta_crudo, "tiempo.npy"), tiempo.astype(np.float32, copy=False))


    log(f"Crudo y tiempo guardados en: {carpeta_crudo}")


def analizar_frecuencia_fft(datos, fs_real, carpeta_fft, sufijo="crudo", logger=None):
    """
    Calcula el espectro de amplitud de la FFT normalizado en µV (espectro unilateral).

    Normalización:
        amp_uV[k] = (2 * |FFT[k]|) / N   para k = 1 … N/2-1  (frecuencias internas)
        amp_uV[0] = |FFT[0]| / N          (componente DC)
        amp_uV[N/2] = |FFT[N/2]| / N      (Nyquist, si N par)

    El factor 2 compensa la pérdida de energía al conservar solo el espectro
    unilateral. Dividir por N elimina la dependencia con la duración del registro,
    dejando la magnitud en las mismas unidades que la señal de entrada (µV).

    Archivos guardados:
        freqs_{sufijo}.npy   → frecuencias en Hz (float32)
        fft_amp_{sufijo}.npy → amplitud en µV, espectro unilateral (float32)
        fft_db_{sufijo}.npy  → misma amplitud en dBµV, solo por compatibilidad
    """
    log = _mklogger(logger)

    n_can, n_muestras = datos.shape
    freqs = np.fft.rfftfreq(n_muestras, d=1.0 / fs_real).astype(np.float32)
    n_freqs = freqs.size

    fft_amp = np.empty((n_can, n_freqs), dtype=np.float32)

    for canal in range(n_can):
        espectro = np.fft.rfft(datos[canal].astype(np.float64))

        # Normalización a amplitud de pico en µV
        amp = np.abs(espectro) / float(n_muestras)

        # Doblar frecuencias internas para conservar amplitud en espectro unilateral
        if n_freqs > 2:
            amp[1:-1] *= 2.0

        fft_amp[canal] = amp.astype(np.float32)

    # dBµV ref 1 µV — conservado por compatibilidad con la interfaz existente
    fft_db = (20.0 * np.log10(np.maximum(fft_amp, 1e-12))).astype(np.float32)

    np.save(os.path.join(carpeta_fft, f"freqs_{sufijo}.npy"), freqs)
    np.save(os.path.join(carpeta_fft, f"fft_amp_{sufijo}.npy"), fft_amp)
    np.save(os.path.join(carpeta_fft, f"fft_db_{sufijo}.npy"), fft_db)

    log(f"FFT {sufijo} guardada.")


def calcular_espectrograma_resumen(datos, fs_real, carpeta_salida, sufijo="filtrado", logger=None):
    """
    Calcula un resumen tiempo-frecuencia por canal usando PSD/STFT.

    La PSD lineal conserva unidades de potencia espectral; si la señal de
    entrada está en µV, la PSD lineal estaría en µV²/Hz. Para visualización
    se transforma a dB, lo cual no es equivalente a la magnitud FFT en µV.
    """
    log = _mklogger(logger)
    x = np.asarray(datos, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("datos debe tener forma (canales, muestras)")
    if x.shape[1] < 2:
        raise ValueError("No hay muestras suficientes para calcular espectrograma.")

    os.makedirs(carpeta_salida, exist_ok=True)

    fs_real = float(fs_real)
    n_canales, n_muestras = x.shape
    nperseg = int(max(2, round(2 * fs_real)))
    noverlap = int(max(0, round(1 * fs_real)))

    if nperseg > n_muestras:
        nperseg = n_muestras
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)


    freqs_guardar = None
    times_guardar = None
    resumen_por_canal = {}

    def _media_banda_db(freqs_eje, sxx_db, fmin, fmax):
        mask = (freqs_eje >= float(fmin)) & (freqs_eje < float(fmax))
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(sxx_db[mask, :]))

    for canal in range(n_canales):
        freqs, times, sxx = spectrogram(
            x[canal],
            fs=fs_real,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )

        mask_f = (freqs >= 1.0) & (freqs <= 40.0)
        freqs_sel = freqs[mask_f]
        sxx_sel = sxx[mask_f, :]

        if freqs_guardar is None:
            freqs_guardar = freqs_sel.astype(np.float32, copy=False)
            times_guardar = times.astype(np.float32, copy=False)

        if freqs_sel.size == 0 or sxx_sel.size == 0:
            resumen_por_canal[f"canal_{canal + 1}"] = {
                "potencia_media_delta_db": float("nan"),
                "potencia_media_theta_db": float("nan"),
                "potencia_media_alfa_db": float("nan"),
                "potencia_media_beta_db": float("nan"),
                "frecuencia_dominante_promedio_hz": float("nan"),
                "potencia_media_global_db": float("nan"),
            }
            continue

        # Se usa dB para mejorar el contraste visual del mapa tiempo-frecuencia.
        sxx_db = 10.0 * np.log10(sxx_sel + 1e-12)

        idx_dom = np.argmax(sxx_db, axis=0)
        frec_dom = float(np.nanmean(freqs_sel[idx_dom])) if idx_dom.size else float("nan")

        resumen_por_canal[f"canal_{canal + 1}"] = {
            "potencia_media_delta_db": _media_banda_db(freqs_sel, sxx_db, *bandas["Delta"]),
            "potencia_media_theta_db": _media_banda_db(freqs_sel, sxx_db, *bandas["Theta"]),
            "potencia_media_alfa_db": _media_banda_db(freqs_sel, sxx_db, *bandas["Alfa"]),
            "potencia_media_beta_db": _media_banda_db(freqs_sel, sxx_db, *bandas["Beta"]),
            "frecuencia_dominante_promedio_hz": frec_dom,
            "potencia_media_global_db": float(np.nanmean(sxx_db)),
        }

    resumen = {
        "metodo": "STFT/spectrogram",
        "escala": "dB",
        "unidad_color": "Potencia espectral (dB)",
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "ventana": "hann",
        "rango_frecuencia_hz": [1, 40],
        "bandas_hz": {b: list(v) for b, v in bandas.items()},
        "resumen_por_canal": resumen_por_canal,
    }

    np.save(os.path.join(carpeta_salida, f"freqs_{sufijo}.npy"), freqs_guardar)
    np.save(os.path.join(carpeta_salida, f"times_{sufijo}.npy"), times_guardar)

    ruta_json = os.path.join(carpeta_salida, f"resumen_espectrograma_{sufijo}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    log(
        f"Resumen de espectrograma guardado en: {ruta_json} "
        f"(canales={n_canales}, nperseg={nperseg}, noverlap={noverlap})"
    )
    return resumen


# =========================
# DETECTOR DE CANALES
# =========================
def _robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return 0.6745 * (x - med) / mad


def detectar_canales_atipicos(datos, fs):
    """
    Detector por criterios:
      - amplitud extrema
      - canal muy bajo/plano
      - exceso de alta frecuencia
      - alfa dominante exagerada
      - espectro distinto al resto
    """
    x = np.asarray(datos, dtype=np.float64)
    n_canales, n_muestras = x.shape

    # =========================
    # Métricas temporales
    # =========================
    stds = np.std(x, axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    ptp = np.ptp(x, axis=1)

    # =========================
    # FFT y potencia espectral
    # =========================
    freqs = np.fft.rfftfreq(n_muestras, d=1.0 / float(fs))
    amp = np.abs(np.fft.rfft(x, axis=1))
    pot = amp ** 2

    idx_delta_detector = (
        (freqs >= float(bandas["Delta"][0])) &
        (freqs <  float(bandas["Delta"][1]))
    )

    idx_lento_amplio = (
        (freqs >= float(bandas["Delta"][0])) &
        (freqs <  float(bandas["Theta"][1]))
    )

    idx_alfa_dominante = (freqs >= 9.0) & (freqs <= 11.0)
    idx_baja_media = (freqs >= 1.0) & (freqs < 20.0)
    idx_alta_frecuencia = (freqs >= 20.0) & (freqs <= 40.0)
    idx_espectro_detector = (freqs >= 1.0) & (freqs <= 40.0)

    energia_delta_detector = np.sum(pot[:, idx_delta_detector], axis=1)
    energia_lento_amplio = np.sum(pot[:, idx_lento_amplio], axis=1)
    energia_alfa_dominante = np.sum(pot[:, idx_alfa_dominante], axis=1)
    energia_baja_media = np.sum(pot[:, idx_baja_media], axis=1)
    energia_alta_frecuencia = np.sum(pot[:, idx_alta_frecuencia], axis=1)
    energia_espectro_detector = np.sum(pot[:, idx_espectro_detector], axis=1)

    ratio_hf = energia_alta_frecuencia / np.maximum(energia_baja_media, 1e-12)
    ratio_alpha_dom = energia_alfa_dominante / np.maximum(energia_lento_amplio, 1e-12)
    ratio_delta = energia_delta_detector / np.maximum(energia_espectro_detector, 1e-12)

    # =========================
    # Comparación del espectro con la mediana
    # =========================
    espectro_norm = pot[:, idx_espectro_detector].copy()
    espectro_norm /= np.maximum(np.sum(espectro_norm, axis=1, keepdims=True), 1e-12)

    espectro_mediano = np.median(espectro_norm, axis=0)
    espectro_mediano /= np.maximum(np.sum(espectro_mediano), 1e-12)

    dist_espectral = np.sqrt(np.sum((espectro_norm - espectro_mediano[None, :]) ** 2, axis=1))

    # =========================
    # Z robustos
    # =========================
    z_std = _robust_z(stds)
    z_rms = _robust_z(rms)
    z_ptp = _robust_z(ptp)
    z_hf = _robust_z(ratio_hf)
    z_alpha = _robust_z(ratio_alpha_dom)
    z_dist = _robust_z(dist_espectral)
    z_delta = _robust_z(ratio_delta)

    # =========================
    # Reglas por tipo
    # =========================
    
    mask_amp_base_sosp = (z_ptp >= 3.2) | (z_rms >= 2.8) | (z_std >= 2.8)

    dominio_delta_claro = (z_delta >= 2.0)

    # amplitud sola no entra como fuerte
    mask_amp_fuerte = np.zeros(n_canales, dtype=bool)

    # solo entra como sospechoso si NO está explicada por delta
    # y además viene acompañada de otra rareza
    mask_amp_sosp = mask_amp_base_sosp & ((~dominio_delta_claro) & ((z_hf >= 1.5) | (z_dist >= 1.5)))

    mask_bajo_fuerte = (z_rms <= -2.8) | (z_std <= -2.8)
    mask_bajo_sosp = (z_rms <= -2.2) | (z_std <= -2.2)

    mask_hf_fuerte = (z_hf >= 3.0) & ((z_rms >= 1.0) | (z_ptp >= 1.0))
    mask_hf_sosp = (z_hf >= 2.2)

    mask_alpha_sosp = (z_alpha >= 2.5)
    mask_alpha_fuerte = (z_alpha >= 3.2) & (z_dist >= 2.0)

    mask_dist_fuerte = (z_dist >= 3.0) & ((z_hf >= 1.5) | (z_alpha >= 1.5) | (z_ptp >= 1.5))
    mask_dist_sosp = (z_dist >= 2.2)

    mask_fuerte = (
        mask_amp_fuerte |
        mask_bajo_fuerte |
        mask_hf_fuerte |
        mask_alpha_fuerte |
        mask_dist_fuerte
    )

    mask_sospechoso = (
        ~mask_fuerte & (
            mask_amp_sosp |
            mask_bajo_sosp |
            mask_hf_sosp |
            mask_alpha_sosp |
            mask_dist_sosp
        )
    )

    idx_fuertes = np.where(mask_fuerte)[0]
    idx_sospechosos = np.where(mask_sospechoso)[0]

    razones = []
    score = np.maximum.reduce([
        np.abs(z_std),
        np.abs(z_rms),
        np.abs(z_ptp),
        np.abs(z_hf),
        np.abs(z_alpha),
        np.abs(z_dist),
    ])

    for i in range(n_canales):
        r = []
        if mask_amp_fuerte[i]:
            r.append("amplitud_extrema")
        elif mask_amp_sosp[i]:
            r.append("amplitud_alta")

        if mask_bajo_fuerte[i]:
            r.append("canal_bajo_fuerte")
        elif mask_bajo_sosp[i]:
            r.append("canal_bajo")

        if mask_hf_fuerte[i]:
            r.append("ruido_hf_fuerte")
        elif mask_hf_sosp[i]:
            r.append("ruido_hf")

        if mask_alpha_fuerte[i]:
            r.append("alfa_dominante_fuerte")
        elif mask_alpha_sosp[i]:
            r.append("alfa_dominante")

        if mask_dist_fuerte[i]:
            r.append("espectro_anomalo_fuerte")
        elif mask_dist_sosp[i]:
            r.append("espectro_anomalo")

        if not r:
            r.append("normal")

        razones.append(" | ".join(r))

    detalle = {
        "std": stds.astype(np.float32),
        "rms": rms.astype(np.float32),
        "ptp": ptp.astype(np.float32),
        "ratio_hf": ratio_hf.astype(np.float32),
        "ratio_alpha_dom": ratio_alpha_dom.astype(np.float32),
        "dist_espectral": dist_espectral.astype(np.float32),
        "ratio_delta": ratio_delta.astype(np.float32),
        "z_delta": z_delta.astype(np.float32),
        "z_std": z_std.astype(np.float32),
        "z_rms": z_rms.astype(np.float32),
        "z_ptp": z_ptp.astype(np.float32),
        "z_hf": z_hf.astype(np.float32),
        "z_alpha": z_alpha.astype(np.float32),
        "z_dist": z_dist.astype(np.float32),
        "score": score.astype(np.float32),
        "razones": np.array(razones, dtype=object),
        "canales_fuertes_idx0": idx_fuertes.astype(np.int32),
        "canales_fuertes_idx1": (idx_fuertes + 1).astype(np.int32),
        "canales_sospechosos_idx0": idx_sospechosos.astype(np.int32),
        "canales_sospechosos_idx1": (idx_sospechosos + 1).astype(np.int32),
    }

    return idx_fuertes, idx_sospechosos, detalle


def guardar_reporte_canales_atipicos(
    carpeta_salida,
    idx_fuertes,
    idx_sospechosos,
    detalle,
    nombre_archivo,
    logger=None
):
    log = _mklogger(logger)
    os.makedirs(carpeta_salida, exist_ok=True)

    np.save(os.path.join(carpeta_salida, "score_canales.npy"), detalle["score"])
    np.save(os.path.join(carpeta_salida, "std_canales.npy"), detalle["std"])
    np.save(os.path.join(carpeta_salida, "rms_canales.npy"), detalle["rms"])
    np.save(os.path.join(carpeta_salida, "ptp_canales.npy"), detalle["ptp"])
    np.save(os.path.join(carpeta_salida, "ratio_hf_canales.npy"), detalle["ratio_hf"])
    np.save(os.path.join(carpeta_salida, "ratio_delta_canales.npy"), detalle["ratio_delta"])
    np.save(os.path.join(carpeta_salida, "z_delta_canales.npy"), detalle["z_delta"])

    np.save(os.path.join(carpeta_salida, "canales_atipicos_fuertes_idx0.npy"), detalle["canales_fuertes_idx0"])
    np.save(os.path.join(carpeta_salida, "canales_atipicos_fuertes_idx1.npy"), detalle["canales_fuertes_idx1"])

    np.save(os.path.join(carpeta_salida, "canales_sospechosos_idx0.npy"), detalle["canales_sospechosos_idx0"])
    np.save(os.path.join(carpeta_salida, "canales_sospechosos_idx1.npy"), detalle["canales_sospechosos_idx1"])

    np.save(os.path.join(carpeta_salida, "ratio_alpha_dom_canales.npy"), detalle["ratio_alpha_dom"])
    np.save(os.path.join(carpeta_salida, "dist_espectral_canales.npy"), detalle["dist_espectral"])
    np.save(os.path.join(carpeta_salida, "z_alpha_canales.npy"), detalle["z_alpha"])
    np.save(os.path.join(carpeta_salida, "z_dist_canales.npy"), detalle["z_dist"])

    ruta_txt = os.path.join(carpeta_salida, "reporte_canales_atipicos.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(f"Archivo: {nombre_archivo}\n")
        f.write("Detector robusto de canales atípicos\n")
        f.write("=" * 50 + "\n\n")

        f.write("Criterio:\n")
        f.write("  Detector por reglas combinadas:\n")
        f.write("  - amplitud extrema\n")
        f.write("  - canal bajo/plano\n")
        f.write("  - ruido de alta frecuencia\n")
        f.write("  - alfa dominante exagerada\n")
        f.write("  - espectro anómalo respecto al resto\n\n")

        if len(idx_fuertes) == 0:
            f.write("No se detectaron canales atípicos fuertes.\n")
        else:
            f.write("Canales atípicos fuertes (base 1):\n")
            f.write(", ".join(str(i + 1) for i in idx_fuertes) + "\n\n")

        if len(idx_sospechosos) == 0:
            f.write("No se detectaron canales sospechosos.\n\n")
        else:
            f.write("Canales sospechosos (base 1):\n")
            f.write(", ".join(str(i + 1) for i in idx_sospechosos) + "\n\n")

        f.write("Detalle por canal fuerte:\n")
        if len(idx_fuertes) == 0:
            f.write("  Ninguno\n")
        else:
            for i in idx_fuertes:
                f.write(
                    f"Canal {i+1:02d} | "
                    f"score={detalle['score'][i]:.2f} | "
                    f"z_std={detalle['z_std'][i]:.2f} | "
                    f"z_rms={detalle['z_rms'][i]:.2f} | "
                    f"z_ptp={detalle['z_ptp'][i]:.2f} | "
                    f"z_hf={detalle['z_hf'][i]:.2f} | "
                    f"z_alpha={detalle['z_alpha'][i]:.2f} | "
                    f"z_delta={detalle['z_delta'][i]:.2f} | "
                    f"z_dist={detalle['z_dist'][i]:.2f} | "
                    f"razon={detalle['razones'][i]}\n"
                )

        f.write("\nDetalle por canal sospechoso:\n")
        if len(idx_sospechosos) == 0:
            f.write("  Ninguno\n")
        else:
            for i in idx_sospechosos:
                f.write(
                    f"Canal {i+1:02d} | "
                    f"score={detalle['score'][i]:.2f} | "
                    f"z_std={detalle['z_std'][i]:.2f} | "
                    f"z_rms={detalle['z_rms'][i]:.2f} | "
                    f"z_ptp={detalle['z_ptp'][i]:.2f} | "
                    f"z_hf={detalle['z_hf'][i]:.2f}\n"
                )

    log(f"Reporte de canales atípicos guardado en: {ruta_txt}")


# =========================
# FILTRADO / ICA / WAVELET
# =========================
def filtro_notch(datos, fs_real, freq=60, Q=30):
    nyquist = 0.5 * fs_real
    notch_freq = freq / nyquist
    b, a = iirnotch(notch_freq, Q)
    datos_filtrados = filtfilt(b, a, datos, axis=1)
    return datos_filtrados


def _frecuencia_cercana(freq, frecuencias, tolerancia_hz=0.5):
    try:
        freq = float(freq)
    except Exception:
        return False

    for valor in frecuencias or []:
        try:
            valor = float(valor)
        except Exception:
            continue
        if abs(freq - valor) < float(tolerancia_hz):
            return True
    return False


def _mask_busqueda_picos(freqs):
    mask = (
        (freqs >= float(AUTO_PICOS_RANGO_BUSQUEDA[0]))
        & (freqs <= float(AUTO_PICOS_RANGO_BUSQUEDA[1]))
    )
    for freq_excluir in AUTO_PICOS_EXCLUIR_FRECUENCIAS:
        try:
            freq_excluir = float(freq_excluir)
        except Exception:
            continue
        mask &= (np.abs(freqs - freq_excluir) > float(AUTO_PICOS_EXCLUSION_CENTRAL_HZ))
    return mask


def _mask_entorno_local(freqs, f0):
    mask = (
        (freqs >= (float(f0) - float(AUTO_PICOS_VENTANA_LOCAL_HZ)))
        & (freqs <= (float(f0) + float(AUTO_PICOS_VENTANA_LOCAL_HZ)))
        & (np.abs(freqs - float(f0)) > float(AUTO_PICOS_EXCLUSION_CENTRAL_HZ))
    )
    for freq_excluir in AUTO_PICOS_EXCLUIR_FRECUENCIAS:
        try:
            freq_excluir = float(freq_excluir)
        except Exception:
            continue
        if abs(float(f0) - freq_excluir) <= float(AUTO_PICOS_VENTANA_LOCAL_HZ):
            mask &= (np.abs(freqs - freq_excluir) > float(AUTO_PICOS_EXCLUSION_CENTRAL_HZ))
    return mask


def _detectar_candidatos_fft_1d(amp_1d, freqs, mask_busqueda, prominence_min):
    idx_busqueda = np.where(mask_busqueda)[0]
    if idx_busqueda.size < 3:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    amp_busqueda = np.asarray(amp_1d[idx_busqueda], dtype=np.float64)
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
    if df <= 0.0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    distancia_bins = max(1, int(round(0.5 / df)))
    ancho_max_bins = max(1, int(round(float(AUTO_PICOS_ANCHO_MAX_HZ) / df)))

    try:
        peaks_rel, props = find_peaks(
            amp_busqueda,
            distance=distancia_bins,
            prominence=max(float(prominence_min), 1e-12),
            width=(None, ancho_max_bins),
        )
    except Exception:
        peaks_rel = np.array([], dtype=int)
        props = {}

    if peaks_rel.size == 0:
        peaks_rel = np.where(
            (amp_busqueda[1:-1] > amp_busqueda[:-2])
            & (amp_busqueda[1:-1] >= amp_busqueda[2:])
            & (amp_busqueda[1:-1] > np.nanmedian(amp_busqueda))
        )[0] + 1
        props = {"prominences": np.zeros(peaks_rel.size, dtype=np.float64)}

    prominencias = np.asarray(props.get("prominences", np.zeros(peaks_rel.size)), dtype=np.float64)
    return idx_busqueda[peaks_rel], prominencias


def atenuar_senoidal_exacta_canales(
    datos,
    fs_real,
    freq,
    canales_idx=None,
    factor=AUTO_PICOS_FACTOR_ATENUACION,
    logger=None
):
    log = _mklogger(logger)
    salida = np.asarray(datos, dtype=np.float64).copy()

    factor = max(0.0, min(1.0, float(factor)))
    if factor <= 0.0:
        return salida

    try:
        freq = float(freq)
    except Exception:
        return salida

    if freq <= 0.0 or freq >= (float(fs_real) / 2.0):
        log(f"[Aviso] Atenuacion omitida: {freq:.1f} Hz fuera del rango valido.")
        return salida

    if salida.ndim != 2 or salida.shape[0] == 0 or salida.shape[1] < 2:
        return salida

    if canales_idx is None:
        idx_validos = np.arange(salida.shape[0], dtype=int)
    else:
        idx_validos = sorted({
            int(i)
            for i in canales_idx
            if isinstance(i, (int, np.integer)) and 0 <= int(i) < salida.shape[0]
        })
        idx_validos = np.asarray(idx_validos, dtype=int)

    if idx_validos.size == 0:
        return salida

    n_muestras = salida.shape[1]
    t = np.arange(n_muestras, dtype=np.float64) / float(fs_real)
    B = np.column_stack([
        np.sin(2.0 * np.pi * float(freq) * t),
        np.cos(2.0 * np.pi * float(freq) * t),
    ])

    try:
        coef, *_ = np.linalg.lstsq(B, salida[idx_validos].T, rcond=None)
    except Exception:
        return salida

    componente = (B @ coef).T
    salida[idx_validos] = salida[idx_validos] - factor * componente
    log(f"[OK] Atenuacion sinusoidal exacta aplicada: {freq:.2f} Hz | factor={factor:.2f}.")
    return salida.astype(np.float64, copy=False)


def atenuar_senoidal_exacta(datos, fs_real, freq, factor=AUTO_PICOS_FACTOR_ATENUACION, logger=None):
    return atenuar_senoidal_exacta_canales(
        datos,
        fs_real,
        freq,
        canales_idx=None,
        factor=factor,
        logger=logger
    )


def detectar_picos_tecnicos_estrechos(datos, fs_real, logger=None):
    log = _mklogger(logger)
    resultado = {
        "frecuencias_aplicar": [],
        "frecuencias_candidatas": [],
        "detalles": [],
        "rango_busqueda_hz": [float(AUTO_PICOS_RANGO_BUSQUEDA[0]), float(AUTO_PICOS_RANGO_BUSQUEDA[1])],
        "factor_atenuacion": float(AUTO_PICOS_FACTOR_ATENUACION),
        "criterio": "pico estrecho en proporcion suficiente de canales",
        "max_frecuencias": int(AUTO_PICOS_MAX_FRECUENCIAS),
    }

    log("[Proceso] Buscando picos tecnicos estrechos entre 4-35 Hz...")

    x = np.asarray(datos, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 4:
        return resultado

    fs_real = float(fs_real)
    if fs_real <= 0:
        return resultado

    n_canales, n_muestras = x.shape
    freqs = np.fft.rfftfreq(n_muestras, d=1.0 / fs_real)
    if freqs.size < 3:
        return resultado

    amp = np.abs(np.fft.rfft(x, axis=1)) / max(int(n_muestras), 1)
    amp_global = np.nanmedian(amp, axis=0)
    mask_busqueda = _mask_busqueda_picos(freqs)
    prominencia_min = max(float(np.nanmedian(amp_global[mask_busqueda])) * 0.10, 1e-12) if np.any(mask_busqueda) else 1e-12
    idx_candidatos, prominencias = _detectar_candidatos_fft_1d(
        amp_global,
        freqs,
        mask_busqueda,
        prominence_min=prominencia_min
    )
    if idx_candidatos.size == 0:
        return resultado

    resultado["frecuencias_candidatas"] = [float(freqs[i]) for i in idx_candidatos]

    pico_delta = None
    pico_delta_global = None
    if AUTO_PICOS_USAR_CRITERIO_DELTA:
        mask_delta = (
            (freqs >= float(AUTO_PICOS_BANDA_DELTA[0]))
            & (freqs <= float(AUTO_PICOS_BANDA_DELTA[1]))
        )
        if np.any(mask_delta):
            pico_delta = np.max(amp[:, mask_delta], axis=1)
            pico_delta = np.maximum(pico_delta, 1e-12)
            pico_delta_global = float(np.nanmax(amp_global[mask_delta]))

    detalles_detectados = []
    for pos, idx_f in enumerate(idx_candidatos):
        f0 = float(freqs[idx_f])
        mask_local = _mask_entorno_local(freqs, f0)
        if not np.any(mask_local):
            continue

        mediana_local_global = float(np.nanmedian(amp_global[mask_local]))
        mediana_local_global = max(mediana_local_global, 1e-12)
        cumple_global_local = amp_global[idx_f] > (float(AUTO_PICOS_FACTOR_LOCAL) * mediana_local_global)
        cumple_global_delta = bool(
            pico_delta_global is not None and amp_global[idx_f] > max(float(pico_delta_global), 1e-12)
        )
        if not (cumple_global_local or cumple_global_delta):
            continue

        mask_canal_local = np.zeros(n_canales, dtype=bool)
        if AUTO_PICOS_USAR_CRITERIO_LOCAL:
            mediana_local = np.nanmedian(amp[:, mask_local], axis=1)
            mediana_local = np.maximum(mediana_local, 1e-12)
            mask_canal_local = amp[:, idx_f] > (float(AUTO_PICOS_FACTOR_LOCAL) * mediana_local)

        mask_canal_delta = np.zeros(n_canales, dtype=bool)
        if AUTO_PICOS_USAR_CRITERIO_DELTA and pico_delta is not None:
            mask_canal_delta = amp[:, idx_f] > pico_delta

        prop_local = float(np.mean(mask_canal_local)) if n_canales else 0.0
        prop_delta = float(np.mean(mask_canal_delta)) if n_canales else 0.0
        pct_local = prop_local * 100.0
        pct_delta = prop_delta * 100.0

        usar_local = AUTO_PICOS_USAR_CRITERIO_LOCAL and (prop_local >= float(AUTO_PICOS_UMBRAL_PROP_CANALES))
        usar_delta = AUTO_PICOS_USAR_CRITERIO_DELTA and (prop_delta >= float(AUTO_PICOS_UMBRAL_PROP_CANALES))

        if not (usar_local or usar_delta):
            continue

        if usar_local and usar_delta:
            criterio_usado = "local+delta"
        elif usar_local:
            criterio_usado = "local"
        else:
            criterio_usado = "delta"

        detalles_detectados.append({
            "frecuencia_hz": f0,
            "proporcion_canales_local": prop_local,
            "porcentaje_canales_local": pct_local,
            "proporcion_canales_delta": prop_delta,
            "porcentaje_canales_delta": pct_delta,
            "criterio_usado": criterio_usado,
            "_severidad": max(prop_local, prop_delta),
            "_prominencia": float(prominencias[pos]) if pos < prominencias.size else 0.0,
            "_porcentaje_afectados": max(pct_local, pct_delta),
        })

    if not detalles_detectados:
        return resultado

    detalles_detectados.sort(
        key=lambda item: (item["_severidad"], item["_prominencia"], item["_porcentaje_afectados"]),
        reverse=True,
    )

    if len(detalles_detectados) > int(AUTO_PICOS_MAX_FRECUENCIAS):
        log(
            f"[Aviso] Se detectaron {len(detalles_detectados)} picos candidatos; "
            f"se corregiran los {int(AUTO_PICOS_MAX_FRECUENCIAS)} mas prominentes."
        )
        detalles_detectados = detalles_detectados[:int(AUTO_PICOS_MAX_FRECUENCIAS)]

    detalles_finales = []
    for detalle in detalles_detectados:
        log(
            f"[OK] Pico tecnico global detectado: {detalle['frecuencia_hz']:.2f} Hz | "
            f"canales afectados: {detalle['_porcentaje_afectados']:.1f} % | "
            f"criterio: {detalle['criterio_usado']}."
        )
        detalles_finales.append({
            "frecuencia_hz": float(detalle["frecuencia_hz"]),
            "proporcion_canales_local": float(detalle["proporcion_canales_local"]),
            "porcentaje_canales_local": float(detalle["porcentaje_canales_local"]),
            "proporcion_canales_delta": float(detalle["proporcion_canales_delta"]),
            "porcentaje_canales_delta": float(detalle["porcentaje_canales_delta"]),
            "criterio_usado": str(detalle["criterio_usado"]),
        })

    resultado["frecuencias_aplicar"] = [float(item["frecuencia_hz"]) for item in detalles_finales]
    resultado["detalles"] = detalles_finales
    return resultado


def aplicar_ica_por_ventanas(datos, fs_real, ventana_seg=10):
    from sklearn.decomposition import FastICA
    muestras_ventana = int(fs_real * ventana_seg)
    n_ventanas = datos.shape[1] // muestras_ventana
    datos_ica = np.zeros_like(datos)

    for i in range(n_ventanas):
        inicio = i * muestras_ventana
        fin = inicio + muestras_ventana
        ventana = datos[:, inicio:fin]

        if ventana.shape[1] < muestras_ventana:
            break

        ica = FastICA(
            n_components=min(64, ventana.shape[0]),
            random_state=42,
            max_iter=2000,
            whiten="unit-variance"
        )

        try:
            componentes = ica.fit_transform(ventana.T)
            reconstruido = ica.inverse_transform(componentes).T
            datos_ica[:, inicio:fin] = reconstruido

        except Exception as e:
            print(f"Error en ventana {i}: {e}")
            datos_ica[:, inicio:fin] = ventana

    fin_procesado = n_ventanas * muestras_ventana
    if fin_procesado < datos.shape[1]:
        datos_ica[:, fin_procesado:] = datos[:, fin_procesado:]

    return datos_ica


def wavelet_denoise_1d(signal_1d, wavelet=WAVELET_NAME, nivel=WAVELET_NIVEL):
    import pywt
    signal_1d = np.array(signal_1d, dtype=np.float32, copy=True)

    coef = pywt.wavedec(signal_1d, wavelet, level=nivel)

    umbral = 0.5 * np.std(coef[-1]) if len(coef[-1]) > 0 else 0.0
    coef[1:] = [pywt.threshold(c, umbral, mode="soft") for c in coef[1:]]

    rec = pywt.waverec(coef, wavelet)

    if rec.size > signal_1d.size:
        rec = rec[:signal_1d.size]
    elif rec.size < signal_1d.size:
        rec = np.pad(rec, (0, signal_1d.size - rec.size), mode="edge")

    return rec.astype(np.float32, copy=False)


def aplicar_wavelet_por_canales(datos, logger=None):
    log = _mklogger(logger)

    datos = np.asarray(datos, dtype=np.float32)
    n_canales, n_muestras = datos.shape
    salida = np.empty((n_canales, n_muestras), dtype=np.float32)

    log("Aplicando wavelet canal por canal...")
    for canal in range(n_canales):
        salida[canal] = wavelet_denoise_1d(datos[canal])
        if (canal + 1) % 8 == 0 or canal == n_canales - 1:
            log(f"Wavelet: canal {canal + 1}/{n_canales}")

    return salida


# =========================
# INFORME
# =========================
def energia_relativa_por_canal(senal_ref, bandas_dict, fs, bandas_def, logger=None):
    log = _mklogger(logger)

    E_total = np.sum(np.asarray(senal_ref, dtype=np.float64) ** 2, axis=1)
    E_total = np.where(E_total == 0, 1e-12, E_total)

    porcentajes = {}
    for b in bandas_def.keys():
        xb = np.asarray(bandas_dict[b], dtype=np.float64)
        Eb = np.sum(xb ** 2, axis=1)
        porcentajes[b] = (Eb / E_total) * 100.0

    promedios = {b: float(np.mean(porcentajes[b])) for b in bandas_def.keys()}
    return porcentajes, promedios


def _limpiar_nombre_canal(nombre: str) -> str:
    if not isinstance(nombre, str):
        return ""
    s = nombre.strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"([._-]?AVG)$", "", s)
    return s


def limpiar_nombre_canal(nombre):
    return _limpiar_nombre_canal(nombre)

def obtener_nombres_canales_64():
    """
    Devuelve la lista fija de canales EEG de NeuroX.
    Evita depender de canales_64.txt y de rutas externas.
    """
    return [limpiar_nombre_canal(c) for c in NOMBRES_CANALES_64]


def normalizar_canales_calidad_por_indice(lista_canales, nombres_canales):
    """
    Convierte CANAL10 / Canal 10 al nombre real según nombres_canales[9].
    Si ya viene como FP1, FZ, CPZ, etc., lo conserva.
    """
    salida = []

    for canal in lista_canales or []:
        simple = limpiar_nombre_canal(canal)

        m = re.match(r"^CANAL(\d+)$", simple)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(nombres_canales):
                simple = limpiar_nombre_canal(nombres_canales[idx])

        if simple:
            salida.append(simple)

    return salida


def _region_de_canal(nombre_simple: str) -> str:
    e = _limpiar_nombre_canal(nombre_simple)

    if not e:
        return "Otro"

    # Si por alguna razón cayó el fallback "Canal X"
    if e.startswith("CANAL"):
        return "Otro"

    if e in ("M1", "M2", "A1", "A2"):
        return "Mastoides"

    # Canales en Z bien ubicados
    if e in ("FPZ", "AFZ", "FZ"):
        return "Frontal"
    if e in ("FCZ", "CZ"):
        return "Central"
    if e in ("CPZ", "PZ"):
        return "Parietal"
    if e in ("POZ", "OZ"):
        return "Occipital"

    if e.startswith(("FP", "AF")):
        return "Frontal"
    if e.startswith("F") and not e.startswith(("FC", "FT", "FP")):
        return "Frontal"

    if e.startswith(("O", "PO", "CB")):
        return "Occipital"

    if e.startswith(("P", "CP")):
        return "Parietal"

    if e.startswith(("T", "FT", "TP")):
        return "Temporal"

    if e.startswith(("C", "FC")):
        return "Central"

    return "Otro"

def energia_relativa_por_region(pct_por_canal: dict, nombres_canales: list, bandas_def: dict, idx_excluir=None):
    """
    Calcula el promedio por región excluyendo canales indicados.
    """
    if idx_excluir is None:
        idx_excluir = []
    idx_excluir = set(int(i) for i in idx_excluir)

    regiones = {}
    for i, nom in enumerate(nombres_canales):
        if i in idx_excluir:
            continue
        reg = _region_de_canal(nom)
        regiones.setdefault(reg, []).append(i)

    promedios_region = {}
    for reg, idxs in regiones.items():
        promedios_region[reg] = {}
        for b in bandas_def.keys():
            v = np.asarray(pct_por_canal[b], dtype=np.float64)
            if len(idxs):
                promedios_region[reg][b] = float(np.nanmean(v[idxs]))
            else:
                promedios_region[reg][b] = float("nan")

    return promedios_region


def calcular_psd_welch_por_canal(datos, fs_real, logger=None):
    """
    Calcula PSD por canal con Welch sobre una señal limpia (canales x muestras).
    """
    log = _mklogger(logger)
    x = np.asarray(datos, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("datos debe tener forma (canales, muestras)")

    n_muestras = x.shape[1]
    if n_muestras < 2:
        raise ValueError("No hay muestras suficientes para calcular Welch.")

    fs_real = float(fs_real)
    nperseg = int(min(max(2, round(fs_real * 2)), n_muestras))
    if nperseg < 2:
        nperseg = n_muestras

    noverlap = nperseg // 2 if nperseg > 2 else 0
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)

    freqs, psd = welch(
        x,
        fs=fs_real,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=1
    )

    log(f"Welch PSD: nperseg={nperseg}, noverlap={noverlap}, canales={x.shape[0]}")
    return freqs.astype(np.float32), psd.astype(np.float32)


def _integrar_psd_por_banda(freqs, psd, fmin, fmax):
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    mask = (freqs >= float(fmin)) & (freqs < float(fmax))

    if np.count_nonzero(mask) >= 2:
        return _trapz(psd[:, mask], freqs[mask], axis=1)

    if np.count_nonzero(mask) == 1:
        return psd[:, mask][:, 0] * max(float(fmax) - float(fmin), 0.0)

    return np.zeros(psd.shape[0], dtype=np.float64)


def calcular_potencia_relativa_welch(
    datos,
    fs_real,
    bandas_def=None,
    rango_total=None,
    logger=None
):
    """
    Calcula potencia absoluta y relativa por banda usando PSD de Welch.
    La potencia total se limita a 1-30 Hz para coherencia clínica.
    """
    if bandas_def is None:
        bandas_def = bandas
    if rango_total is None:
        rango_total = RANGO_TOTAL_WELCH

    freqs, psd = calcular_psd_welch_por_canal(datos, fs_real, logger=logger)
    pot_total = _integrar_psd_por_banda(freqs, psd, rango_total[0], rango_total[1])
    pot_total_safe = np.where(pot_total <= 0, 1e-12, pot_total)

    pot_abs = {}
    pot_rel = {}
    for banda, (fmin, fmax) in bandas_def.items():
        abs_banda = _integrar_psd_por_banda(freqs, psd, fmin, fmax)
        pot_abs[banda] = abs_banda.astype(np.float32)
        pot_rel[banda] = ((abs_banda / pot_total_safe) * 100.0).astype(np.float32)

    promedios_globales = {
        b: float(np.nanmean(np.asarray(pot_rel[b], dtype=np.float64)))
        for b in bandas_def.keys()
    }

    return {
        "freqs": freqs,
        "psd": psd,
        "pot_total": pot_total.astype(np.float32),
        "pot_abs": pot_abs,
        "pot_rel": pot_rel,
        "promedios_globales": promedios_globales,
    }


def guardar_potencia_relativa_welch(
    carpeta_salida,
    resultado_welch,
    nombres_canales=None,
    bandas_def=None,
    idx_excluir=None,
    logger=None
):
    """
    Guarda el análisis Welch en cache para uso de la interfaz.
    """
    log = _mklogger(logger)
    if bandas_def is None:
        bandas_def = bandas
    if idx_excluir is None:
        idx_excluir = []

    os.makedirs(carpeta_salida, exist_ok=True)

    np.save(os.path.join(carpeta_salida, "welch_freqs.npy"), resultado_welch["freqs"])
    np.save(os.path.join(carpeta_salida, "welch_psd.npy"), resultado_welch["psd"])
    np.save(os.path.join(carpeta_salida, "pot_total_welch.npy"), resultado_welch["pot_total"])
    np.savez(
        os.path.join(carpeta_salida, "pot_abs_welch.npz"),
        **{b: resultado_welch["pot_abs"][b].astype(np.float32, copy=False) for b in bandas_def.keys()}
    )
    np.savez(
        os.path.join(carpeta_salida, "pot_rel_welch.npz"),
        **{b: resultado_welch["pot_rel"][b].astype(np.float32, copy=False) for b in bandas_def.keys()}
    )

    promedios_region = {}
    if nombres_canales is not None:
        promedios_region = energia_relativa_por_region(
            resultado_welch["pot_rel"],
            nombres_canales,
            bandas_def,
            idx_excluir=idx_excluir
        )

    resumen = {
        "metodo": "Welch",
        "rango_total_hz": list(RANGO_TOTAL_WELCH),
        "bandas_hz": {b: list(v) for b, v in bandas_def.items()},
        "promedios_globales": resultado_welch["promedios_globales"],
        "promedios_regionales": promedios_region,
    }

    with open(os.path.join(carpeta_salida, "pot_rel_promedios.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    log(f"Potencia relativa guardada en: {carpeta_salida}")
    return resumen


def _sumar_bandas_para_ratio(pot_rel, bandas_ratio):
    total = None
    for b in bandas_ratio:
        if b not in pot_rel:
            raise KeyError(f"Falta banda para ratio Welch: {b}")
        arr = np.asarray(pot_rel[b], dtype=np.float64)
        total = arr.copy() if total is None else total + arr
    if total is None:
        raise ValueError("Ratio Welch sin bandas definidas.")
    return total


def calcular_ratios_welch_principales(pot_rel, ratios_def=None, eps=1e-12):
    """
    Calcula proporciones principales desde potencia relativa ya calculada.
    No recalcula PSD ni bandas.
    """
    if ratios_def is None:
        ratios_def = RATIOS_WELCH_PRINCIPALES

    ratios = {}
    for nombre, spec in ratios_def.items():
        num = _sumar_bandas_para_ratio(pot_rel, spec["numerador"])
        den = _sumar_bandas_para_ratio(pot_rel, spec["denominador"])
        den_safe = np.where(np.abs(den) <= eps, eps, den)
        ratios[nombre] = (num / den_safe).astype(np.float32)

    promedios_globales = {
        nombre: float(np.nanmean(np.asarray(valores, dtype=np.float64)))
        for nombre, valores in ratios.items()
    }

    return {
        "ratios": ratios,
        "promedios_globales": promedios_globales,
    }


def guardar_ratios_welch_principales(
    carpeta_salida,
    resultado_ratios,
    nombres_canales=None,
    ratios_def=None,
    idx_excluir=None,
    logger=None
):
    """
    Guarda las 4 proporciones principales Welch en cache.
    """
    log = _mklogger(logger)
    if ratios_def is None:
        ratios_def = RATIOS_WELCH_PRINCIPALES
    if idx_excluir is None:
        idx_excluir = []

    os.makedirs(carpeta_salida, exist_ok=True)

    ratios = resultado_ratios["ratios"]
    np.savez(
        os.path.join(carpeta_salida, "ratios_welch_principales.npz"),
        **{k: ratios[k].astype(np.float32, copy=False) for k in ratios_def.keys()}
    )

    promedios_region = {}
    if nombres_canales is not None:
        promedios_region = energia_relativa_por_region(
            ratios,
            nombres_canales,
            {k: None for k in ratios_def.keys()},
            idx_excluir=idx_excluir
        )

    resumen = {
        "metodo": "ratios_desde_potencia_relativa_welch",
        "ratios": {
            k: {
                "label": v["label"],
                "numerador": list(v["numerador"]),
                "denominador": list(v["denominador"]),
            }
            for k, v in ratios_def.items()
        },
        "promedios_globales": resultado_ratios["promedios_globales"],
        "promedios_regionales": promedios_region,
    }

    with open(os.path.join(carpeta_salida, "ratios_promedios_principales.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    log(f"Ratios Welch principales guardados en: {carpeta_salida}")
    return resumen


def calcular_snr_por_canal(datos, fs, banda_senal=(1, 30), banda_ruido=(40, 100)):
    """
    Calcula la relación señal/ruido estimada por canal usando Welch.
    """
    snr_lineal = []
    snr_db = []

    for canal in range(datos.shape[0]):
        x = np.asarray(datos[canal], dtype=float)

        if x.size < 2:
            snr_lineal.append(0.0)
            snr_db.append(-120.0)
            continue

        nperseg = min(int(fs * 2), x.size)
        nperseg = max(2, int(nperseg))

        f, pxx = welch(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2
        )

        idx_senal = (f >= banda_senal[0]) & (f <= banda_senal[1])
        idx_ruido = (f >= banda_ruido[0]) & (f <= banda_ruido[1])

        potencia_senal = _trapz(pxx[idx_senal], f[idx_senal]) if np.any(idx_senal) else 0.0
        potencia_ruido = _trapz(pxx[idx_ruido], f[idx_ruido]) if np.any(idx_ruido) else 0.0

        potencia_ruido = max(float(potencia_ruido), 1e-12)
        potencia_senal = max(float(potencia_senal), 1e-12)

        snr = potencia_senal / potencia_ruido
        snr_lineal.append(float(snr))
        snr_db.append(float(10 * np.log10(snr)))

    snr_lineal = np.asarray(snr_lineal, dtype=float)
    snr_db = np.asarray(snr_db, dtype=float)

    resumen_snr = {
        "banda_senal_hz": list(banda_senal),
        "banda_ruido_hz": list(banda_ruido),
        "snr_global_lineal": float(np.mean(snr_lineal)),
        "snr_global_db": float(np.mean(snr_db)),
        "snr_mediana_db": float(np.median(snr_db)),
        "snr_min_db": float(np.min(snr_db)),
        "snr_max_db": float(np.max(snr_db)),
    }

    return resumen_snr, snr_db


def interpretar_snr(snr_db):
    if snr_db >= 10:
        return "Buena relación señal/ruido. La señal útil predomina sobre el ruido de fondo."
    elif snr_db >= 5:
        return "Relación señal/ruido aceptable. Existe ruido moderado, pero la señal conserva utilidad analítica."
    elif snr_db >= 0:
        return "Relación señal/ruido baja. La señal presenta ruido importante y debe interpretarse con precaución."
    else:
        return "Ruido dominante. La muestra puede no ser técnicamente confiable para interpretación."


def clasificar_calidad_global(
    porcentaje_atipicos,
    porcentaje_sospechosos,
    porcentaje_ponderado,
    snr_global_db=None
):
    try:
        porcentaje_atipicos = float(porcentaje_atipicos)
    except Exception:
        porcentaje_atipicos = 0.0
    try:
        porcentaje_sospechosos = float(porcentaje_sospechosos)
    except Exception:
        porcentaje_sospechosos = 0.0
    try:
        porcentaje_ponderado = float(porcentaje_ponderado)
    except Exception:
        porcentaje_ponderado = 0.0

    if porcentaje_atipicos > 30 or porcentaje_ponderado > 45:
        estado = "Deficiente"
    elif porcentaje_atipicos > 20 or porcentaje_ponderado > 30:
        estado = "Comprometida"
    elif porcentaje_atipicos > 5 or porcentaje_ponderado > 15:
        estado = "Aceptable"
    else:
        estado = "Buena"

    try:
        snr_valida = float(snr_global_db)
    except Exception:
        snr_valida = None

    if snr_valida is not None and np.isfinite(snr_valida):
        if snr_valida < 0:
            if estado in ("Buena", "Aceptable"):
                estado = "Comprometida"
        elif snr_valida < 3:
            if estado == "Buena":
                estado = "Aceptable"

    return estado


def interpretar_estado_global(
    estado_global,
    porcentaje_ponderado,
    snr_global_db,
    region_predominante,
    porcentaje_atipicos=None,
    porcentaje_sospechosos=None
):
    try:
        porcentaje_ponderado = float(porcentaje_ponderado)
    except Exception:
        porcentaje_ponderado = 0.0
    try:
        snr_global_db = float(snr_global_db)
    except Exception:
        snr_global_db = float("nan")

    detalle_componentes = ""
    try:
        pct_atipicos = float(porcentaje_atipicos)
        pct_sospechosos = float(porcentaje_sospechosos)
        detalle_componentes = (
            f" Los canales atípicos fuertes representan {pct_atipicos:.1f}% y los sospechosos "
            f"{pct_sospechosos:.1f}%."
        )
    except Exception:
        pass

    if estado_global == "Buena":
        return (
            "La adquisición presenta baja proporción de canales con indicadores técnicos atípicos. "
            f"El índice técnico ponderado es {porcentaje_ponderado:.1f}% y la SNR global estimada es "
            f"{snr_global_db:.2f} dB.{detalle_componentes} "
            "La clasificación es orientativa y debe complementarse con inspección visual."
        )

    if estado_global == "Aceptable":
        return (
            "La adquisición presenta algunos canales con indicadores técnicos atípicos o sospechosos, "
            f"pero la proporción ponderada se mantiene en un rango aceptable para revisión "
            f"({porcentaje_ponderado:.1f}%). La SNR global estimada es {snr_global_db:.2f} dB."
            f"{detalle_componentes} La clasificación es técnica y orientativa."
        )

    if estado_global == "Comprometida":
        return (
            "La adquisición presenta una proporción relevante de canales atípicos fuertes o "
            f"ponderación técnica elevada ({porcentaje_ponderado:.1f}%). La SNR global estimada es "
            f"{snr_global_db:.2f} dB.{detalle_componentes} "
            f"Se recomienda revisar visualmente los canales marcados, especialmente en la región "
            f"{region_predominante}."
        )

    return (
        "La adquisición presenta una proporción alta de canales atípicos fuertes o ponderación técnica "
        f"muy elevada ({porcentaje_ponderado:.1f}%). La SNR global estimada es {snr_global_db:.2f} dB."
        f"{detalle_componentes} Se recomienda revisar la adquisición antes de interpretar los resultados."
    )


def analizar_distribucion_regional(canales_afectados, regiones=REGIONES_EEG_64):
    """
    Cuenta cuántos canales afectados hay por región.
    """
    canales_limpios = [limpiar_nombre_canal(c) for c in canales_afectados]

    conteo = {region: 0 for region in regiones}

    for canal in canales_limpios:
        for region, lista_canales in regiones.items():
            if canal in lista_canales:
                conteo[region] += 1
                break

    region_predominante = max(conteo, key=conteo.get) if conteo else "No determinada"
    max_canales = conteo.get(region_predominante, 0)

    if max_canales == 0:
        region_predominante = "Sin predominio regional"
        interpretacion = "No se observa una concentración regional clara de canales afectados."
    elif region_predominante == "Frontal":
        interpretacion = (
            "La afectación predominante en región frontal puede asociarse con parpadeo, "
            "movimiento ocular, actividad facial o mala impedancia en electrodos anteriores."
        )
    elif region_predominante == "Frontocentral":
        interpretacion = (
            "La afectación frontocentral puede relacionarse con tensión muscular frontal, "
            "movimiento o problemas de contacto en electrodos anteriores y centrales."
        )
    elif region_predominante == "Central/Temporal":
        interpretacion = (
            "La afectación central o temporal puede asociarse con actividad muscular, "
            "movimiento mandibular, tensión lateral o problemas de contacto en zonas temporales."
        )
    elif region_predominante == "Temporo-parietal":
        interpretacion = (
            "La afectación temporo-parietal puede relacionarse con movimiento, tensión muscular lateral "
            "o contacto deficiente en electrodos posteriores laterales."
        )
    elif region_predominante == "Parietal":
        interpretacion = (
            "La afectación parietal puede sugerir problemas de contacto en electrodos posteriores "
            "o contaminación por movimiento."
        )
    elif region_predominante == "Occipital":
        interpretacion = (
            "La afectación occipital puede asociarse con mala impedancia posterior, movimiento, "
            "apoyo de cabeza o artefactos localizados en regiones posteriores."
        )
    elif region_predominante == "Referencia/Periféricos":
        interpretacion = (
            "La afectación en canales de referencia o periféricos puede alterar la estabilidad global "
            "de la señal y debe revisarse antes del análisis."
        )
    else:
        interpretacion = (
            "Los canales afectados se distribuyen en varias regiones, lo que puede sugerir ruido "
            "generalizado o problemas durante la adquisición."
        )

    return {
        "conteo_regional": conteo,
        "region_predominante": region_predominante,
        "interpretacion_regional": interpretacion
    }


def crear_grafico_circular_calidad(resumen, ruta_salida):
    valores = [
        resumen["n_normales"],
        resumen["n_sospechosos"],
        resumen["n_atipicos"]
    ]

    etiquetas = [
        f"Normales\n{resumen['porcentaje_normales']:.1f}%",
        f"Sospechosos\n{resumen['porcentaje_sospechosos']:.1f}%",
        f"Atípicos\n{resumen['porcentaje_atipicos']:.1f}%"
    ]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=130)

    if sum(valores) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        ax.axis("off")
    else:
        ax.pie(
            valores,
            labels=etiquetas,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Distribución de calidad por canal")

    fig.tight_layout()
    fig.savefig(ruta_salida, bbox_inches="tight")
    plt.close(fig)


def calcular_resumen_calidad_senal(
    datos_para_snr,
    fs,
    nombres_canales,
    canales_atipicos,
    canales_sospechosos,
    carpeta_cache,
    logger=None
):
    """
    Calcula y guarda el resumen de calidad técnica de la señal.
    `datos_para_snr` debe preservar contenido de alta frecuencia si se va a
    estimar ruido en 40-100 Hz; si la señal ya fue limitada a 40 Hz, la SNR
    quedará artificialmente inflada.
    """
    log = _mklogger(logger)

    carpeta_calidad = os.path.join(carpeta_cache, "calidad_senal")
    os.makedirs(carpeta_calidad, exist_ok=True)

    canales_atipicos = [limpiar_nombre_canal(c) for c in canales_atipicos]
    canales_sospechosos = [limpiar_nombre_canal(c) for c in canales_sospechosos]
    canales_atipicos = normalizar_canales_calidad_por_indice(
        canales_atipicos,
        nombres_canales
    )

    canales_sospechosos = normalizar_canales_calidad_por_indice(
        canales_sospechosos,
        nombres_canales
    )

    n_canales_total = int(len(nombres_canales))
    n_atipicos = int(len(canales_atipicos))
    n_sospechosos = int(len(canales_sospechosos))
    n_normales = max(n_canales_total - n_atipicos - n_sospechosos, 0)

    peso_sospechoso = 0.5
    porcentaje_atipicos = (n_atipicos / n_canales_total) * 100 if n_canales_total else 0.0
    porcentaje_sospechosos = (n_sospechosos / n_canales_total) * 100 if n_canales_total else 0.0
    porcentaje_normales = (n_normales / n_canales_total) * 100 if n_canales_total else 0.0
    porcentaje_afectado_sin_ponderar = (
        ((n_atipicos + n_sospechosos) / n_canales_total) * 100 if n_canales_total else 0.0
    )
    porcentaje_ponderado = porcentaje_atipicos + peso_sospechoso * porcentaje_sospechosos
    porcentaje_afectado = porcentaje_ponderado

    log("[Proceso] Calculando SNR desde señal cruda original...")
    resumen_snr, snr_db = calcular_snr_por_canal(datos_para_snr, fs)
    snr_global_db = resumen_snr["snr_global_db"]
    log(f"[OK] SNR global estimada desde crudo: {snr_global_db:.2f} dB.")

    canales_afectados = list(dict.fromkeys(canales_atipicos + canales_sospechosos))
    resumen_regional = analizar_distribucion_regional(canales_afectados)

    log("[Proceso] Evaluando calidad tecnica de la adquisicion...")
    log(f"[OK] Canales atipicos fuertes: {n_atipicos}/{n_canales_total} ({porcentaje_atipicos:.1f} %).")
    log(f"[OK] Canales sospechosos: {n_sospechosos}/{n_canales_total} ({porcentaje_sospechosos:.1f} %).")
    log(f"[OK] Indice tecnico ponderado: {porcentaje_ponderado:.1f} %.")

    estado_global = clasificar_calidad_global(
        porcentaje_atipicos=porcentaje_atipicos,
        porcentaje_sospechosos=porcentaje_sospechosos,
        porcentaje_ponderado=porcentaje_ponderado,
        snr_global_db=snr_global_db
    )
    interpretacion_global = interpretar_estado_global(
        estado_global=estado_global,
        porcentaje_ponderado=porcentaje_ponderado,
        snr_global_db=snr_global_db,
        region_predominante=resumen_regional["region_predominante"],
        porcentaje_atipicos=porcentaje_atipicos,
        porcentaje_sospechosos=porcentaje_sospechosos
    )
    interpretacion_snr = interpretar_snr(snr_global_db)
    log(f"[OK] Estado global de calidad tecnica: {estado_global}.")

    resumen = {
        "n_canales_total": n_canales_total,
        "n_normales": n_normales,
        "n_sospechosos": n_sospechosos,
        "n_atipicos": n_atipicos,
        "peso_sospechoso": peso_sospechoso,
        "porcentaje_normales": porcentaje_normales,
        "porcentaje_sospechosos": porcentaje_sospechosos,
        "porcentaje_atipicos": porcentaje_atipicos,
        "porcentaje_afectado_sin_ponderar": porcentaje_afectado_sin_ponderar,
        "porcentaje_ponderado": porcentaje_ponderado,
        "porcentaje_afectado": porcentaje_afectado,
        "canales_atipicos": canales_atipicos,
        "canales_sospechosos": canales_sospechosos,
        "estado_global": estado_global,
        "snr": resumen_snr,
        "snr_fuente": "señal cruda original antes de filtros, notch, atenuaciones, ICA o Wavelet",
        "interpretacion_snr": interpretacion_snr,
        "analisis_regional": resumen_regional,
        "interpretacion_global": interpretacion_global,
        "criterio_clasificacion": {
            "descripcion": (
                "Clasificacion tecnica basada en porcentaje de canales atipicos fuertes "
                "y porcentaje ponderado de atipicos + sospechosos."
            ),
            "formula_ponderada": "porcentaje_ponderado = porcentaje_atipicos + 0.5 * porcentaje_sospechosos",
            "umbrales": {
                "Deficiente": "porcentaje_atipicos > 30 o porcentaje_ponderado > 45",
                "Comprometida": "porcentaje_atipicos > 20 o porcentaje_ponderado > 30",
                "Aceptable": "porcentaje_atipicos > 5 o porcentaje_ponderado > 15",
                "Buena": "resto"
            },
            "snr": "La SNR solo degrada el estado si es < 3 dB; nunca mejora la clasificacion."
        },
        "ecuaciones": {
            "porcentaje_afectado": "porcentaje_afectado = porcentaje_ponderado",
            "porcentaje_ponderado": "% ponderado = % atipicos + 0.5 * % sospechosos",
            "porcentaje_afectado_sin_ponderar": "% sin ponderar = ((N_atipicos + N_sospechosos) / N_total) * 100",
            "snr_lineal": "SNR = P_senal / P_ruido",
            "snr_db": "SNR_dB = 10 * log10(SNR)",
            "potencia_senal": "P_senal = potencia integrada en 1-30 Hz",
            "potencia_ruido": "P_ruido = potencia integrada en 40-100 Hz"
        }
    }

    ruta_json = os.path.join(carpeta_calidad, "resumen_calidad.json")
    ruta_snr = os.path.join(carpeta_calidad, "snr_por_canal.npy")
    ruta_png = os.path.join(carpeta_calidad, "grafico_calidad.png")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    np.save(ruta_snr, np.asarray(snr_db, dtype=np.float32))
    crear_grafico_circular_calidad(resumen, ruta_png)

    log(f"Resumen de calidad guardado en: {ruta_json}")
    return resumen




def evaluar_banda(banda, valor, umbral_amarillo=5):
    low, high = rangos_referencia[banda]

    if valor < low:
        estado = "Por debajo"
        diff_pp = low - valor
        detalle = f"{estado} por {diff_pp:.2f} pp"
    elif valor > high:
        estado = "Por encima"
        diff_pp = valor - high
        detalle = f"{estado} por {diff_pp:.2f} pp"
    else:
        estado = "En rango"
        dist_pp = min(valor - low, high - valor)
        detalle = f"{estado} (a {dist_pp:.2f} pp del límite más cercano)"

    if low <= valor <= high:
        color = "green"
    else:
        if abs(valor - high) <= umbral_amarillo or abs(valor - low) <= umbral_amarillo:
            color = "yellow"
        else:
            color = "red"

    return {
        "banda": banda,
        "valor": float(valor),
        "low": float(low),
        "high": float(high),
        "estado": estado,
        "detalle": detalle,
        "color": color
    }


def interpretar_region(nombre_region, valores_region, promedios_globales):
    """
    Analisis descriptivo por region comparando contra el promedio global
    del mismo registro.
    """
    lineas = []

    for banda in ["Delta", "Theta", "Alfa", "Beta"]:
        vr = valores_region.get(banda, np.nan)
        vg = promedios_globales.get(banda, np.nan)

        if np.isnan(vr) or np.isnan(vg):
            continue

        diff = vr - vg

        if abs(diff) < 3:
            continue

        if diff > 0:
            lineas.append(
                f"{banda}: {vr:.1f}% en la region, por encima del promedio global ({vg:.1f}%)."
            )
        else:
            lineas.append(
                f"{banda}: {vr:.1f}% en la region, por debajo del promedio global ({vg:.1f}%)."
            )

    conclusion = ""

    delta_r = valores_region.get("Delta", np.nan)
    alfa_r = valores_region.get("Alfa", np.nan)
    beta_r = valores_region.get("Beta", np.nan)

    delta_g = promedios_globales.get("Delta", np.nan)
    alfa_g = promedios_globales.get("Alfa", np.nan)
    beta_g = promedios_globales.get("Beta", np.nan)

    if nombre_region == "Frontal":
        if not np.isnan(beta_r) and not np.isnan(beta_g) and beta_r >= beta_g + 3:
            conclusion = (
                "Esto sugiere un predominio relativo de actividad rapida en la region frontal."
            )

    elif nombre_region in ("Parietal", "Occipital"):
        if not np.isnan(alfa_r) and not np.isnan(alfa_g) and alfa_r >= alfa_g + 3:
            conclusion = (
                "Esto sugiere un predominio relativo de alfa en region posterior."
            )
        elif not np.isnan(delta_r) and not np.isnan(delta_g) and delta_r >= delta_g + 4:
            conclusion = (
                "Esto describe una mayor proporcion relativa de actividad lenta en esta region posterior."
            )

    elif nombre_region == "Temporal":
        if not np.isnan(delta_r) and not np.isnan(delta_g) and delta_r >= delta_g + 4:
            conclusion = (
                "Esto sugiere una lentificacion relativa temporal respecto al promedio global del registro."
            )

    elif nombre_region == "Central":
        conclusion = (
            "Los porcentajes regionales se mantienen cercanos al promedio global, sin predominio marcado."
        )

    if not lineas:
        return (
            f"{nombre_region}: los porcentajes de energia relativa regional son cercanos "
            f"al promedio global del registro, sin predominio marcado."
        )

    texto = f"{nombre_region}: " + " ".join(lineas)
    if conclusion:
        texto += " " + conclusion

    return texto


REGIONES_PRINCIPALES_INFORME = ("Frontal", "Central", "Temporal", "Parietal", "Occipital")

RATIOS_WELCH_LABELS = {
    "Theta_Alfa": "Theta/Alfa",
    "Delta_Alfa": "Delta/Alfa",
    "Theta_Beta": "Theta/Beta",
    "Lentificacion": "Lentificación",
}


def _leer_json_si_existe(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _cargar_npz_dict(ruta):
    if not os.path.exists(ruta):
        return None
    with np.load(ruta, allow_pickle=False) as z:
        return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


def _fmt_pct_informe(valor):
    try:
        valor = float(valor)
    except Exception:
        return "--"
    if not np.isfinite(valor):
        return "--"
    return f"{valor:.1f} %"


def _fmt_num_informe(valor, decimales=2):
    try:
        valor = float(valor)
    except Exception:
        return "--"
    if not np.isfinite(valor):
        return "--"
    return f"{valor:.{decimales}f}"


def _texto_lista_canales(canales):
    canales = [str(c).strip() for c in (canales or []) if str(c).strip()]
    return ", ".join(canales) if canales else "ninguno"


def _proyectar_dentro_circulo_informe(x, y, radio=1.0, margen=0.985):
    d2 = x * x + y * y
    lim2 = (margen * radio) * (margen * radio)
    if d2 <= lim2 or d2 == 0:
        return x, y
    d = np.sqrt(d2)
    escala = (margen * radio) / d
    return x * escala, y * escala


def _s_por_prefijo_1010_informe(prefijo):
    prefijo = prefijo.upper()
    if prefijo == "FP":
        return 0.10
    if prefijo == "AF":
        return 0.20
    if prefijo == "F":
        return 0.31
    if prefijo in ("FC", "FT"):
        return 0.40
    if prefijo in ("C", "T"):
        return 0.50
    if prefijo in ("CP", "TP"):
        return 0.60
    if prefijo == "P":
        return 0.70
    if prefijo == "PO":
        return 0.80
    if prefijo == "O":
        return 0.90
    return 0.50


def _lateral_por_numero_informe(prefijo, numero):
    prefijo = prefijo.upper()
    if numero in (1, 2):
        lateral = 0.22
    elif numero in (3, 4):
        lateral = 0.42
    elif numero in (5, 6):
        lateral = 0.62
    elif numero in (7, 8):
        lateral = 0.92
    else:
        lateral = 0.50

    if prefijo in ("T", "FT", "TP") and numero in (7, 8):
        lateral = 0.95
    if prefijo in ("FP", "O") and numero in (1, 2):
        lateral = 0.44
    return lateral


def _t_uniforme_en_fila_informe(prefijo, sufijo):
    prefijo = prefijo.upper()
    sufijo = sufijo.upper()
    orden_por_fila = {
        "F": ["7", "5", "3", "1", "Z", "2", "4", "6", "8"],
        "FC": ["5", "3", "1", "Z", "2", "4", "6"],
        "C": ["5", "3", "1", "Z", "2", "4", "6"],
        "CP": ["5", "3", "1", "Z", "2", "4", "6"],
        "P": ["7", "5", "3", "1", "Z", "2", "4", "6", "8"],
        "PO": ["7", "5", "3", "Z", "4", "6", "8"],
    }
    if prefijo not in orden_por_fila:
        return None
    orden = orden_por_fila[prefijo]
    if sufijo not in orden:
        return None
    indice = orden.index(sufijo)
    medio = (len(orden) - 1) / 2.0
    return (indice - medio) / medio


def _factor_apertura_fila_informe(prefijo):
    prefijo = prefijo.upper()
    if prefijo in ("FP", "O"):
        return 1.15
    if prefijo in ("C", "FC", "CP"):
        return 0.82
    if prefijo in ("F", "P"):
        return 0.92
    if prefijo == "PO":
        return 1.00
    if prefijo in ("T", "FT", "TP"):
        return 0.97
    return 1.0


def _coord_1010_curvada_norm_informe(etiqueta):
    etiqueta = (etiqueta or "").upper().strip()

    if etiqueta == "M1":
        x, y = _proyectar_dentro_circulo_informe(-1.0, 0.18)
        return x, -y
    if etiqueta == "M2":
        x, y = _proyectar_dentro_circulo_informe(1.0, 0.18)
        return x, -y
    if etiqueta == "CB1":
        x, y = _proyectar_dentro_circulo_informe(-0.43, 0.86)
        return x, -y
    if etiqueta == "CB2":
        x, y = _proyectar_dentro_circulo_informe(0.43, 0.86)
        return x, -y

    match = re.match(r"^([A-Z]+)(\d+|Z)$", etiqueta)
    if not match:
        return None

    prefijo, sufijo = match.group(1), match.group(2)
    s = _s_por_prefijo_1010_informe(prefijo)
    y_base = (s - 0.5) * 2.0
    if prefijo == "F":
        y_base -= 0.02

    x_max = np.sqrt(max(0.0, 1.0 - y_base * y_base)) * 0.98
    if x_max < 1e-6:
        return None

    t = _t_uniforme_en_fila_informe(prefijo, sufijo)
    apertura = _factor_apertura_fila_informe(prefijo)
    lateral_global = 0.90
    if prefijo in ("T", "FT", "TP"):
        lateral_global = 0.94
    if prefijo in ("C", "FC", "CP"):
        lateral_global = 0.82

    if sufijo == "Z":
        x = 0.0
        y = y_base
    elif t is not None:
        x = t * x_max * lateral_global * apertura
        y = y_base
    else:
        numero = int(sufijo)
        lateral = _lateral_por_numero_informe(prefijo, numero) * lateral_global
        signo = -1 if (numero % 2 == 1) else 1
        x = signo * lateral * x_max
        y = y_base

    if s < 0.50:
        signo_curva = -1
    elif s > 0.50:
        signo_curva = 1
    else:
        signo_curva = 0

    if prefijo in ("F", "P"):
        amplitud = 0.14
    elif prefijo in ("FC", "CP", "FT", "TP"):
        amplitud = 0.10
    else:
        amplitud = 0.08

    xn = x / x_max
    if prefijo == "F":
        y = y + signo_curva * amplitud * 0.75 * (xn * xn)
    else:
        y = y + signo_curva * amplitud * (xn * xn)

    x, y = _proyectar_dentro_circulo_informe(x, y)
    return x, -y


def _posiciones_topograficas_por_canal(nombres_canales):
    xs, ys, etiquetas, indices = [], [], [], []
    for i, nombre in enumerate(nombres_canales):
        simple = _limpiar_nombre_canal(nombre)
        posicion = _coord_1010_curvada_norm_informe(simple)
        if posicion is None:
            continue
        xs.append(float(posicion[0]))
        ys.append(float(posicion[1]))
        etiquetas.append(simple)
        indices.append(i)

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        etiquetas,
        np.asarray(indices, dtype=np.int32),
    )


def _dibujar_cabeza_topomap(ax, linewidth=1.5, zorder=10):
    cabeza = plt.Circle((0, 0), 0.985, fill=False, color="black", linewidth=linewidth, zorder=zorder)
    ax.add_patch(cabeza)
    ax.plot(
        [-0.09, 0.0, 0.09],
        [0.985, 1.10, 0.985],
        color="black",
        lw=max(1.1, linewidth - 0.15),
        clip_on=False,
        zorder=zorder,
    )

    t = np.linspace(-0.9, 0.9, 80)
    ax.plot(
        -1.01 - 0.015 * np.cos(t),
        0.22 * np.sin(t),
        color="black",
        lw=max(1.1, linewidth - 0.15),
        clip_on=False,
        zorder=zorder,
    )
    ax.plot(
        1.01 + 0.015 * np.cos(t),
        0.22 * np.sin(t),
        color="black",
        lw=max(1.1, linewidth - 0.15),
        clip_on=False,
        zorder=zorder,
    )

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.15, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def _crear_grid_topomap_informe(resolucion=250, radio=1.0):
    gx = np.linspace(-1.05, 1.05, resolucion)
    gy = np.linspace(-1.05, 1.05, resolucion)
    xi, yi = np.meshgrid(gx, gy)
    mascara = (xi ** 2 + yi ** 2) <= radio ** 2
    return xi, yi, mascara


def _interpolar_topomap_informe(xs, ys, zs, xi, yi, mascara):
    puntos = np.column_stack([xs, ys])

    try:
        zi_cubic = griddata(points=puntos, values=zs, xi=(xi, yi), method="cubic")
    except Exception:
        zi_cubic = None

    if zi_cubic is None or np.all(np.isnan(zi_cubic)):
        try:
            zi_cubic = griddata(points=puntos, values=zs, xi=(xi, yi), method="linear")
        except Exception:
            zi_cubic = None

    zi_nearest = griddata(points=puntos, values=zs, xi=(xi, yi), method="nearest")
    if zi_cubic is None:
        zi = zi_nearest
    else:
        zi = np.where(np.isnan(zi_cubic), zi_nearest, zi_cubic)

    zi = gaussian_filter(np.asarray(zi, dtype=np.float64), sigma=2.2)
    zi[~mascara] = np.nan
    return zi


def _valores_canales_para_topomap(valores, nombres_canales):
    xs, ys, etiquetas, indices = _posiciones_topograficas_por_canal(nombres_canales)
    if indices.size == 0:
        return None

    arr = np.asarray(valores, dtype=np.float64).ravel()
    if arr.size <= int(np.max(indices)):
        return None

    zs = arr[indices]
    validos = np.isfinite(zs)
    if np.count_nonzero(validos) < 4:
        return None

    return {
        "xs": xs[validos],
        "ys": ys[validos],
        "zs": zs[validos],
        "etiquetas": [etiquetas[i] for i, ok in enumerate(validos) if ok],
    }


def crear_mapa_calidad_1010(resumen_calidad, nombres_canales, ruta_salida):
    if not isinstance(resumen_calidad, dict):
        return False

    atipicos = {limpiar_nombre_canal(c) for c in resumen_calidad.get("canales_atipicos", [])}
    sospechosos = {limpiar_nombre_canal(c) for c in resumen_calidad.get("canales_sospechosos", [])}

    fig, ax = plt.subplots(figsize=(5.15, 5.05), dpi=200)
    try:
        _dibujar_cabeza_topomap(ax, linewidth=1.45, zorder=8)

        dibujados = []
        for nombre in nombres_canales:
            simple = limpiar_nombre_canal(nombre)
            posicion = _coord_1010_curvada_norm_informe(simple)
            if posicion is None:
                continue
            x, y = posicion
            if simple in atipicos:
                color = "#DC2626"
                size = 42
            elif simple in sospechosos:
                color = "#F59E0B"
                size = 32
            else:
                color = "#22C55E"
                size = 28

            ax.scatter([x], [y], s=size, c=color, edgecolors="black", linewidths=0.55, zorder=9)
            dibujados.append((simple, x, y))

        for simple, x, y in dibujados:
            ax.text(x, y + 0.021, simple, fontsize=4.6, ha="center", va="bottom", color="#1F2937")

        legend_handles = [
            Patch(facecolor="#22C55E", edgecolor="black", label="Normal"),
            Patch(facecolor="#F59E0B", edgecolor="black", label="Sospechoso"),
            Patch(facecolor="#DC2626", edgecolor="black", label="Atípico"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            fontsize=7.1,
            frameon=False
        )
        fig.tight_layout(pad=0.45)
        fig.savefig(ruta_salida, dpi=200, bbox_inches="tight", pad_inches=0.04)
        return True
    finally:
        plt.close(fig)


def _promedios_regionales_principales(resumen):
    if not isinstance(resumen, dict):
        return {}

    regionales = resumen.get("promedios_regionales", {})
    if not isinstance(regionales, dict):
        return {}

    salida = {}
    for region in REGIONES_PRINCIPALES_INFORME:
        valores = regionales.get(region)
        if isinstance(valores, dict):
            salida[region] = valores
    return salida


def _region_maxima_para_clave(resumen, clave):
    regionales = _promedios_regionales_principales(resumen)
    mejor_region = None
    mejor_valor = None
    for region in REGIONES_PRINCIPALES_INFORME:
        try:
            valor = float(regionales.get(region, {}).get(clave, np.nan))
        except Exception:
            continue
        if not np.isfinite(valor):
            continue
        if mejor_valor is None or valor > mejor_valor:
            mejor_region = region
            mejor_valor = valor
    return mejor_region, mejor_valor


def _region_lenta_destacada(resumen_welch, umbral=2.0):
    regionales = _promedios_regionales_principales(resumen_welch)
    puntajes = {}
    for region, valores in regionales.items():
        try:
            delta = float(valores.get("Delta", np.nan))
            theta = float(valores.get("Theta", np.nan))
        except Exception:
            continue
        if not np.isfinite(delta) or not np.isfinite(theta):
            continue
        puntajes[region] = (delta + theta) / 2.0

    if len(puntajes) < 2:
        return None

    region_max = max(puntajes, key=puntajes.get)
    region_min = min(puntajes, key=puntajes.get)
    if (puntajes[region_max] - puntajes[region_min]) < umbral:
        return None
    return region_max


def _region_alfa_baja_destacada(resumen_welch, umbral=2.0):
    regionales = _promedios_regionales_principales(resumen_welch)
    alfas = {}
    for region, valores in regionales.items():
        try:
            alfa = float(valores.get("Alfa", np.nan))
        except Exception:
            continue
        if np.isfinite(alfa):
            alfas[region] = alfa

    if len(alfas) < 2:
        return None

    region_min = min(alfas, key=alfas.get)
    region_max = max(alfas, key=alfas.get)
    if (alfas[region_max] - alfas[region_min]) < umbral:
        return None
    return region_min


def _metricas_calidad_resumen(resumen_calidad):
    if not isinstance(resumen_calidad, dict):
        return None

    pct_ponderado = resumen_calidad.get(
        "porcentaje_ponderado",
        resumen_calidad.get("porcentaje_afectado", np.nan)
    )
    pct_atipicos = resumen_calidad.get("porcentaje_atipicos", np.nan)
    pct_sospechosos = resumen_calidad.get("porcentaje_sospechosos", np.nan)

    return {
        "estado": resumen_calidad.get("estado_global", "--"),
        "pct_ponderado": _fmt_pct_informe(pct_ponderado),
        "pct_atipicos": _fmt_pct_informe(pct_atipicos),
        "pct_sospechosos": _fmt_pct_informe(pct_sospechosos),
        "snr_db": _fmt_num_informe(resumen_calidad.get("snr", {}).get("snr_global_db", np.nan)),
        "region": resumen_calidad.get("analisis_regional", {}).get("region_predominante", "No determinada"),
        "atipicos": _texto_lista_canales(resumen_calidad.get("canales_atipicos", [])),
        "sospechosos": _texto_lista_canales(resumen_calidad.get("canales_sospechosos", [])),
    }


def _registro_para_informe(nombre_archivo):
    base = os.path.basename(str(nombre_archivo or "")).strip()
    if not base:
        return "Desconocido"
    return os.path.splitext(base)[0]


def buscar_logo_neurox():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(base_dir, "assets", "logo_neurox_horizontal.png"),
        os.path.join(base_dir, "assets", "nombre_neurox.png"),
        os.path.join(base_dir, "logo_neurox_horizontal.png"),
        os.path.join(base_dir, "nombre_neurox.png"),
        os.path.join(base_dir, "assets", "icono_neurox.png"),
        os.path.join(base_dir, "icono_neurox.png"),
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    return None


def _texto_calidad_tecnica_compacto_informe(resumen_calidad):
    metricas = _metricas_calidad_resumen(resumen_calidad)
    if metricas is None:
        return "No se encontró resumen de calidad técnica para este archivo."

    return {
        "rows": [
            ["Estado global", metricas["estado"]],
            ["Canales atípicos fuertes", metricas["pct_atipicos"]],
            ["Canales sospechosos", metricas["pct_sospechosos"]],
            ["Índice técnico ponderado", metricas["pct_ponderado"]],
            ["SNR global estimada", f"{metricas['snr_db']} dB" if metricas["snr_db"] != "--" else "--"],
            ["Región predominante afectada", metricas["region"]],
        ],
        "texto_canales": (
            f"Canales atípicos fuertes detectados: {metricas['atipicos']}\n"
            f"Canales sospechosos detectados: {metricas['sospechosos']}\n\n"
            "Los canales sospechosos se ponderan con menor peso que los atípicos fuertes porque representan "
            "alertas moderadas, no alteraciones técnicas robustas.\n\n"
            "Los canales marcados se reportan como referencia técnica. Su revisión debe complementarse "
            "con inspección visual de la señal."
        ),
    }


def crear_grafico_circular_calidad_informe(resumen, ruta_salida):
    valores = [
        resumen.get("n_normales", 0),
        resumen.get("n_sospechosos", 0),
        resumen.get("n_atipicos", 0),
    ]
    etiquetas = [
        f"Normales\n{float(resumen.get('porcentaje_normales', 0.0)):.1f}%",
        f"Sospechosos\n{float(resumen.get('porcentaje_sospechosos', 0.0)):.1f}%",
        f"Atípicos\n{float(resumen.get('porcentaje_atipicos', 0.0)):.1f}%",
    ]

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
    try:
        if sum(valores) == 0:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.axis("off")
        else:
            ax.pie(
                valores,
                labels=etiquetas,
                colors=["#22C55E", "#F59E0B", "#DC2626"],
                startangle=90,
                autopct=None,
                labeldistance=1.08,
                radius=0.84,
                wedgeprops={"width": 0.38, "edgecolor": "white"},
                textprops={"fontsize": 8},
            )
            ax.text(0, 0, "Calidad\nEEG", ha="center", va="center", fontsize=10, weight="bold")

        fig.tight_layout(pad=0.35)
        fig.savefig(ruta_salida, bbox_inches="tight", pad_inches=0.04)
    finally:
        plt.close(fig)


def crear_topomaps_2x2_informe(
    valores_por_mapa,
    nombres_canales,
    ruta_salida,
    titulos,
    orden=None,
    cmap="turbo",
    cbar_label="%"
):
    if not isinstance(valores_por_mapa, dict) or not valores_por_mapa:
        return False

    if orden is None:
        orden = list(valores_por_mapa.keys())

    fig = plt.figure(figsize=(10.2, 8.2), dpi=220)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 0.055, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        left=0.04,
        right=0.93,
        bottom=0.06,
        top=0.92,
        wspace=0.20,
        hspace=0.24,
    )

    posiciones_axes = [(0, 0), (0, 2), (1, 0), (1, 2)]
    posiciones_caxes = [(0, 1), (0, 3), (1, 1), (1, 3)]
    axes = []
    caxes = []
    for pos_ax, pos_cax in zip(posiciones_axes, posiciones_caxes):
        axes.append(fig.add_subplot(gs[pos_ax[0], pos_ax[1]]))
        caxes.append(fig.add_subplot(gs[pos_cax[0], pos_cax[1]]))

    exito = False
    try:
        for i, clave in enumerate(orden[:4]):
            ax = axes[i]
            cax = caxes[i]
            titulo = titulos.get(clave, clave) if isinstance(titulos, dict) else str(clave)

            if clave not in valores_por_mapa:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                cax.axis("off")
                continue

            datos = _valores_canales_para_topomap(valores_por_mapa[clave], nombres_canales)
            if datos is None:
                ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                cax.axis("off")
                continue

            mappable = None

            try:
                xi, yi, mascara = _crear_grid_topomap_informe()
                zi = _interpolar_topomap_informe(datos["xs"], datos["ys"], datos["zs"], xi, yi, mascara)
                vmin = float(np.nanmin(zi))
                vmax = float(np.nanmax(zi))
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    raise ValueError("Interpolación sin rango válido.")
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6

                niveles = np.linspace(vmin, vmax, 24)
                mappable = ax.contourf(xi, yi, zi, levels=niveles, cmap=cmap)
                ax.contour(xi, yi, zi, levels=10, colors="k", linewidths=0.18, alpha=0.18)
            except Exception:
                vmin = float(np.nanmin(datos["zs"]))
                vmax = float(np.nanmax(datos["zs"]))
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6
                mappable = ax.scatter(
                    datos["xs"],
                    datos["ys"],
                    c=datos["zs"],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=115,
                    edgecolors="none",
                    zorder=2,
                )

            ax.scatter(datos["xs"], datos["ys"], s=12, c="black", edgecolors="white", linewidths=0.18, zorder=11)
            _dibujar_cabeza_topomap(ax, linewidth=1.55, zorder=10)
            ax.set_title(str(titulo), fontsize=10)
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.16)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            cbar = fig.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=6, length=2)
            cbar.set_label(cbar_label, fontsize=7)
            cbar.outline.set_linewidth(0.6)
            exito = True

        for j in range(min(len(orden), 4), 4):
            axes[j].axis("off")
            caxes[j].axis("off")

        fig.savefig(ruta_salida, dpi=220, bbox_inches="tight", pad_inches=0.04)
        return exito
    finally:
        plt.close(fig)


def _tabla_regional_potencia_informe(resumen_welch):
    if not isinstance(resumen_welch, dict):
        return []

    regionales = resumen_welch.get("promedios_regionales", {}) or {}
    filas = []
    for region in REGIONES_PRINCIPALES_INFORME:
        valores = regionales.get(region, {})
        filas.append([
            region,
            _fmt_pct_informe(valores.get("Delta", np.nan)),
            _fmt_pct_informe(valores.get("Theta", np.nan)),
            _fmt_pct_informe(valores.get("Alfa", np.nan)),
            _fmt_pct_informe(valores.get("Beta", np.nan)),
        ])
    return filas


def _descripcion_individual_bandas_informe(resumen_welch):
    textos = []
    for banda in ("Delta", "Theta", "Alfa", "Beta"):
        region_max, _ = _region_maxima_para_clave(resumen_welch, banda)
        region_txt = region_max or "sin predominio regional claro"
        if banda == "Delta":
            textos.append(
                f"Delta: se observa mayor potencia relativa en región {region_txt}. "
                "Este patrón describe predominio de actividad lenta en esa zona."
            )
        elif banda == "Theta":
            textos.append(
                f"Theta: se observa mayor potencia relativa en región {region_txt}. "
                "Esta banda describe actividad lenta y debe interpretarse junto con el estado de alerta, "
                "artefactos y contexto clínico."
            )
        elif banda == "Alfa":
            textos.append(
                f"Alfa: se observa mayor potencia relativa en región {region_txt}. "
                "En registros de reposo, la distribución posterior de alfa suele ser un patrón esperado; "
                "una distribución diferente debe revisarse de forma descriptiva."
            )
        else:
            textos.append(
                f"Beta: se observa mayor potencia relativa en región {region_txt}. "
                "Este predominio puede relacionarse con actividad rápida, tensión muscular o características "
                "propias del registro."
            )
    return "\n\n".join(textos)


def _descripcion_individual_ratios_informe(resumen_ratios):
    textos = []
    for clave in ("Theta_Alfa", "Delta_Alfa", "Theta_Beta", "Lentificacion"):
        region_max, _ = _region_maxima_para_clave(resumen_ratios, clave)
        region_txt = region_max or "sin predominio regional claro"
        if clave == "Theta_Alfa":
            textos.append(
                f"Theta/Alfa: el mayor valor se observa en región {region_txt}. "
                "Esta proporción describe el balance entre actividad lenta theta y actividad alfa."
            )
        elif clave == "Delta_Alfa":
            textos.append(
                f"Delta/Alfa: el mayor valor se observa en región {region_txt}. "
                "Valores elevados describen mayor peso relativo de actividad delta frente a alfa."
            )
        elif clave == "Theta_Beta":
            textos.append(
                f"Theta/Beta: el mayor valor se observa en región {region_txt}. "
                "Esta proporción describe el balance entre actividad lenta theta y actividad rápida beta."
            )
        else:
            textos.append(
                f"Lentificación: el mayor valor se observa en región {region_txt}. "
                "Esta métrica resume el peso relativo de bandas lentas frente a bandas alfa-beta."
            )
    return "\n\n".join(textos)


def construir_resumen_descriptivo_final_informe(resumen_welch, resumen_ratios):
    promedios_welch = resumen_welch.get("promedios_globales", {}) if isinstance(resumen_welch, dict) else {}
    promedios_ratios = resumen_ratios.get("promedios_globales", {}) if isinstance(resumen_ratios, dict) else {}

    banda_global = None
    valores_bandas = {}
    for banda in ("Delta", "Theta", "Alfa", "Beta"):
        try:
            valor = float(promedios_welch.get(banda, np.nan))
        except Exception:
            continue
        if np.isfinite(valor):
            valores_bandas[banda] = valor
    if valores_bandas:
        banda_global = max(valores_bandas, key=valores_bandas.get)

    frases = []
    if banda_global is not None:
        frases.append(f"En conjunto, el análisis muestra un predominio relativo de {banda_global} a nivel global.")

    region_lenta = _region_lenta_destacada(resumen_welch)
    if region_lenta is not None:
        frases.append(f"Regionalmente, se observan mayores valores de actividad lenta en {region_lenta}.")

    region_alfa_baja = _region_alfa_baja_destacada(resumen_welch)
    if region_alfa_baja is not None:
        frases.append(f"La potencia relativa de alfa es comparativamente menor en {region_alfa_baja}.")

    theta_alfa = _fmt_num_informe(promedios_ratios.get("Theta_Alfa", np.nan))
    lentificacion = _fmt_num_informe(promedios_ratios.get("Lentificacion", np.nan))
    if theta_alfa != "--" or lentificacion != "--":
        frases.append(
            f"Las proporciones globales Theta/Alfa ({theta_alfa}) y Lentificación ({lentificacion}) "
            "describen el balance entre actividad lenta y actividad alfa-beta."
        )

    frases.append("Los valores son descriptivos y requieren correlación con la evaluación clínica e inspección visual.")
    return " ".join(frases)


class InformeEEG(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_y(self.t_margin)
            return
        self.set_y(8)
        self.set_font("Helvetica", "", 8.4)
        self.set_text_color(92, 102, 116)
        self.cell(0, 4.5, "NeuroX | Informe qEEG", ln=True, align="R")
        self.set_draw_color(33, 63, 104)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3.2)
        self.set_text_color(28, 32, 38)

    def footer(self):
        self.set_y(-10)
        self.set_draw_color(205, 212, 220)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", "", 8.5)
        registro = getattr(self, "registro_titulo", "Desconocido")
        self.set_text_color(92, 102, 116)
        self.cell(0, 4.5, f"{registro} | Página {self.page_no()}", align="C")
        self.set_text_color(28, 32, 38)

    def ensure_space(self, needed=35):
        if self.get_y() > (self.page_break_trigger - needed):
            self.add_page()

    def add_section_title(self, titulo):
        self.ensure_space(18)
        if self.get_y() > (self.t_margin + 6):
            self.ln(3.5)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(33, 63, 104)
        self.cell(0, 7, str(titulo), ln=True)
        self.set_draw_color(33, 63, 104)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3.2)
        self.set_text_color(28, 32, 38)


    def add_block_title(self, titulo):
        """Título breve para tablas o figuras dentro de una sección."""
        self.ensure_space(12)
        self.set_font("Helvetica", "B", 10.8)
        self.set_text_color(33, 63, 104)
        self.cell(0, 5.8, str(titulo), ln=True)
        self.set_text_color(28, 32, 38)
        self.ln(1.0)

    def add_wrapped_text(self, texto, font_size=10.5):
        self.set_font("Helvetica", "", font_size)
        self.set_text_color(28, 32, 38)
        self.multi_cell(0, 5.8, str(texto))
        self.ln(3.2)

    def add_visual_table(
        self,
        headers,
        rows,
        col_widths=None,
        alignments=None,
        row_height=6.6,
        table_width=None,
        center=True
    ):
        if not headers:
            return

        self.ensure_space(18 + row_height * (len(rows) + 1))
        usable_width = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            total_width = float(table_width) if table_width is not None else usable_width
            col_widths = [total_width / len(headers)] * len(headers)
        if alignments is None:
            alignments = ["C"] * len(headers)

        total_width = sum(col_widths)
        if center:
            x_start = self.l_margin + max(0, (usable_width - total_width) / 2.0)
        else:
            x_start = self.l_margin

        self.set_fill_color(229, 236, 245)
        self.set_draw_color(173, 183, 197)
        self.set_text_color(33, 63, 104)
        self.set_font("Helvetica", "B", 10)

        self.set_x(x_start)
        for header, width in zip(headers, col_widths):
            self.cell(width, row_height, str(header), border=1, align="C", fill=True)
        self.ln(row_height)

        self.set_text_color(28, 32, 38)
        self.set_font("Helvetica", "", 10)
        for row in rows:
            self.set_x(x_start)
            for i, width in enumerate(col_widths):
                align = alignments[i] if i < len(alignments) else "C"
                valor = row[i] if i < len(row) else ""
                self.cell(width, row_height, str(valor), border=1, align=align)
            self.ln(row_height)
        self.ln(4.4)

    def _image_height_mm(self, image_path, width_mm):
        try:
            with Image.open(image_path) as img:
                px_w, px_h = img.size
            if px_w <= 0:
                return float(width_mm)
            return float(width_mm) * float(px_h) / float(px_w)
        except Exception:
            return float(width_mm)

    def add_centered_image(
        self,
        image_path,
        width_mm,
        caption=None,
        needed_height_mm=90,
        gap_after_image=2.0,
        gap_after_caption=1.2,
        gap_no_caption=2.8
    ):
        if not image_path or not os.path.exists(image_path):
            return False
        self.ensure_space(needed_height_mm)
        x = max(self.l_margin, (self.w - width_mm) / 2.0)
        self.image(image_path, x=x, w=width_mm)
        self.ln(gap_after_image)
        if caption:
            self.set_font("Helvetica", "", 8.4)
            self.set_text_color(92, 102, 116)
            self.multi_cell(0, 4.4, str(caption), align="C")
            self.set_text_color(28, 32, 38)
            self.ln(gap_after_caption)
        else:
            self.ln(gap_no_caption)
        return True

    def add_two_images_row(
        self,
        img_left,
        img_right,
        w_left=65,
        w_right=78,
        caption_left=None,
        caption_right=None,
        gap=12,
        needed_height_mm=88
    ):
        existe_izq = bool(img_left and os.path.exists(img_left))
        existe_der = bool(img_right and os.path.exists(img_right))

        if not existe_izq and not existe_der:
            return False
        if existe_izq and not existe_der:
            return self.add_centered_image(
                img_left,
                width_mm=w_left,
                caption=caption_left,
                needed_height_mm=needed_height_mm
            )
        if existe_der and not existe_izq:
            return self.add_centered_image(
                img_right,
                width_mm=w_right,
                caption=caption_right,
                needed_height_mm=needed_height_mm
            )

        h_left = self._image_height_mm(img_left, w_left)
        h_right = self._image_height_mm(img_right, w_right)
        caption_h = 8.4 if (caption_left or caption_right) else 0.0
        block_h = max(h_left, h_right) + caption_h + 5.0
        self.ensure_space(max(needed_height_mm, block_h + 8.0))

        usable_width = self.w - self.l_margin - self.r_margin
        total_width = float(w_left) + float(gap) + float(w_right)
        x_left = self.l_margin + max(0.0, (usable_width - total_width) / 2.0)
        x_right = x_left + float(w_left) + float(gap)
        y_top = self.get_y()

        self.image(img_left, x=x_left, y=y_top, w=w_left)
        self.image(img_right, x=x_right, y=y_top, w=w_right)

        y_caption = y_top + max(h_left, h_right) + 1.8
        self.set_font("Helvetica", "", 8.3)
        self.set_text_color(92, 102, 116)
        if caption_left:
            self.set_xy(x_left, y_caption)
            self.multi_cell(w_left, 4.2, str(caption_left), align="C")
        if caption_right:
            self.set_xy(x_right, y_caption)
            self.multi_cell(w_right, 4.2, str(caption_right), align="C")
        self.set_text_color(28, 32, 38)
        self.set_y(y_caption + caption_h)
        self.ln(2.2)
        return True


def copiar_informe_a_carpeta_externa(ruta_pdf_cache, carpeta_base_origen, logger=None):
    log = _mklogger(logger)
    if not ruta_pdf_cache or not os.path.exists(ruta_pdf_cache):
        return ruta_pdf_cache

    try:
        carpeta_base = os.path.abspath(str(carpeta_base_origen or CARPETA_BASE))
        carpeta_destino = os.path.join(carpeta_base, "INFORMES_NEUROX")
        os.makedirs(carpeta_destino, exist_ok=True)
        ruta_destino = os.path.join(carpeta_destino, os.path.basename(ruta_pdf_cache))
        shutil.copy2(ruta_pdf_cache, ruta_destino)
        log("[OK] Informe guardado en carpeta INFORMES_NEUROX.")
        return ruta_destino
    except Exception:
        log("[Aviso] No se pudo copiar el informe a INFORMES_NEUROX; se conserva en cache.")
        return ruta_pdf_cache


def generar_informe_desde_cache(carpeta_archivo_cache, logger=None):
    log = _mklogger(logger)
    meta_path = os.path.join(carpeta_archivo_cache, "meta.txt")
    archivo_original = "desconocido"
    carpeta_base_origen = CARPETA_BASE
    n_canales_meta = n_canales
    fs_meta = fs
    muestras = None
    duracion_s = None

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("archivo="):
                    archivo_original = line.split("=", 1)[1].strip()
                elif line.startswith("carpeta_base="):
                    carpeta_base_origen = line.split("=", 1)[1].strip()
                elif line.startswith("n_canales="):
                    n_canales_meta = int(line.split("=", 1)[1].strip())
                elif line.startswith("fs="):
                    fs_meta = int(float(line.split("=", 1)[1].strip()))
                elif line.startswith("muestras="):
                    muestras = int(line.split("=", 1)[1].strip())
                elif line.startswith("duracion_s="):
                    duracion_s = float(line.split("=", 1)[1].strip())

    registro_titulo = _registro_para_informe(archivo_original)

    nombres_canales = obtener_nombres_canales_64()

    ruta_tiempo = os.path.join(carpeta_archivo_cache, "crudo", "tiempo.npy")
    tiempo = None
    if os.path.exists(ruta_tiempo):
        tiempo = np.load(ruta_tiempo, mmap_mode="r")
        if muestras is None:
            muestras = int(tiempo.shape[0])
        if duracion_s is None:
            duracion_s = float(tiempo.shape[0] / float(fs_meta)) if fs_meta else 0.0
    if duracion_s is None:
        duracion_s = 0.0

    datos_ica = None
    datos_wavelet = None
    datos_fil = None
    bandas_filtradas = {}
    bandas_filtradas_fil = {}
    try:
        ruta_ica = os.path.join(carpeta_archivo_cache, "ica", "eeg_ica.npy")
        ruta_wavelet = os.path.join(carpeta_archivo_cache, "wavelet", "eeg_wavelet.npy")
        carpeta_bandas = os.path.join(carpeta_archivo_cache, "bandas")
        ruta_fil = os.path.join(carpeta_archivo_cache, "filtrado", "eeg_filtrado.npy")
        carpeta_bandas_fil = os.path.join(carpeta_archivo_cache, "bandas_filtrado")

        if (
            os.path.exists(ruta_ica)
            and os.path.exists(ruta_wavelet)
            and os.path.exists(ruta_fil)
            and os.path.isdir(carpeta_bandas)
            and os.path.isdir(carpeta_bandas_fil)
        ):
            datos_ica = np.load(ruta_ica, mmap_mode="r")
            datos_wavelet = np.load(ruta_wavelet, mmap_mode="r")
            datos_fil = np.load(ruta_fil, mmap_mode="r")

            for banda in bandas.keys():
                ruta_b = os.path.join(carpeta_bandas, f"{banda}.npy")
                ruta_bf = os.path.join(carpeta_bandas_fil, f"{banda}.npy")
                if not os.path.exists(ruta_b) or not os.path.exists(ruta_bf):
                    raise FileNotFoundError(f"Falta banda para actualizar caches de energía relativa: {banda}")
                bandas_filtradas[banda] = np.load(ruta_b, mmap_mode="r")
                bandas_filtradas_fil[banda] = np.load(ruta_bf, mmap_mode="r")

            pct_fil, _prom_fil = energia_relativa_por_canal(
                datos_fil, bandas_filtradas_fil, fs_meta, bandas, logger=logger
            )
            pct_final, _prom_final = energia_relativa_por_canal(
                datos_wavelet, bandas_filtradas, fs_meta, bandas, logger=logger
            )

            carpeta_er_fil = os.path.join(carpeta_archivo_cache, "energia_relativa_filtrado")
            carpeta_er_ica = os.path.join(carpeta_archivo_cache, "energia_relativa_ica")
            carpeta_energia = os.path.join(carpeta_archivo_cache, "energia_relativa")
            os.makedirs(carpeta_er_fil, exist_ok=True)
            os.makedirs(carpeta_er_ica, exist_ok=True)
            os.makedirs(carpeta_energia, exist_ok=True)

            for banda in bandas.keys():
                np.save(os.path.join(carpeta_er_fil, f"{banda}.npy"), pct_fil[banda].astype(np.float32, copy=False))
                np.save(os.path.join(carpeta_er_ica, f"{banda}.npy"), pct_final[banda].astype(np.float32, copy=False))
                np.save(os.path.join(carpeta_energia, f"{banda}.npy"), pct_final[banda].astype(np.float32, copy=False))
    except Exception as e:
        log(f"No pude actualizar los caches de energía relativa para el informe: {e}")

    nombre_limpio = os.path.basename(carpeta_archivo_cache)
    carpeta_informe = os.path.join(carpeta_archivo_cache, "informe")
    carpeta_graficos = os.path.join(carpeta_informe, "graficos")
    carpeta_analisis_welch = os.path.join(carpeta_archivo_cache, "analisis_welch")
    carpeta_calidad = os.path.join(carpeta_archivo_cache, "calidad_senal")
    os.makedirs(carpeta_informe, exist_ok=True)
    os.makedirs(carpeta_graficos, exist_ok=True)

    resumen_calidad = _leer_json_si_existe(os.path.join(carpeta_calidad, "resumen_calidad.json"))
    pot_rel_welch = _cargar_npz_dict(os.path.join(carpeta_analisis_welch, "pot_rel_welch.npz")) or {}
    resumen_welch = _leer_json_si_existe(os.path.join(carpeta_analisis_welch, "pot_rel_promedios.json")) or {}
    ratios_welch = _cargar_npz_dict(os.path.join(carpeta_analisis_welch, "ratios_welch_principales.npz")) or {}
    resumen_ratios = _leer_json_si_existe(
        os.path.join(carpeta_analisis_welch, "ratios_promedios_principales.json")
    ) or {}

    ruta_grafico_calidad = os.path.join(carpeta_calidad, "grafico_calidad.png")
    ruta_mapa_calidad = os.path.join(carpeta_calidad, "mapa_calidad_1010.png")
    if isinstance(resumen_calidad, dict):
        try:
            crear_grafico_circular_calidad_informe(resumen_calidad, ruta_grafico_calidad)
        except Exception as e:
            log(f"No pude generar grafico_calidad.png: {e}")
        try:
            ok_mapa = crear_mapa_calidad_1010(resumen_calidad, nombres_canales, ruta_mapa_calidad)
            if not ok_mapa and os.path.exists(ruta_mapa_calidad):
                os.remove(ruta_mapa_calidad)
        except Exception as e:
            log(f"No pude generar mapa_calidad_1010.png: {e}")

    ruta_topomaps_welch = os.path.join(carpeta_graficos, "topomaps_potencia_welch.png")
    ok_topomaps_welch = False
    if pot_rel_welch:
        try:
            ok_topomaps_welch = crear_topomaps_2x2_informe(
                {b: pot_rel_welch[b] for b in ("Delta", "Theta", "Alfa", "Beta") if b in pot_rel_welch},
                nombres_canales,
                ruta_topomaps_welch,
                {"Delta": "Delta", "Theta": "Theta", "Alfa": "Alfa", "Beta": "Beta"},
                orden=["Delta", "Theta", "Alfa", "Beta"],
                cmap="turbo",
                cbar_label="%",
            )
        except Exception as e:
            log(f"No pude generar topomaps_potencia_welch.png: {e}")
    if (not ok_topomaps_welch) and os.path.exists(ruta_topomaps_welch):
        try:
            os.remove(ruta_topomaps_welch)
        except Exception:
            pass

    ruta_topomaps_ratios = os.path.join(carpeta_graficos, "topomaps_ratios_welch.png")
    ok_topomaps_ratios = False
    if ratios_welch:
        try:
            ok_topomaps_ratios = crear_topomaps_2x2_informe(
                {
                    k: ratios_welch[k]
                    for k in ("Theta_Alfa", "Delta_Alfa", "Theta_Beta", "Lentificacion")
                    if k in ratios_welch
                },
                nombres_canales,
                ruta_topomaps_ratios,
                RATIOS_WELCH_LABELS,
                orden=["Theta_Alfa", "Delta_Alfa", "Theta_Beta", "Lentificacion"],
                cmap="turbo",
                cbar_label="ratio",
            )
        except Exception as e:
            log(f"No pude generar topomaps_ratios_welch.png: {e}")
    if (not ok_topomaps_ratios) and os.path.exists(ruta_topomaps_ratios):
        try:
            os.remove(ruta_topomaps_ratios)
        except Exception:
            pass

    pdf = InformeEEG()
    pdf.registro_titulo = registro_titulo
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    logo_path = buscar_logo_neurox()
    if logo_path is None:
        log("Logo NeuroX no encontrado; se genera informe sin logo.")
    else:
        nombre_logo = os.path.basename(logo_path).lower()
        if "logo_neurox_horizontal" in nombre_logo:
            ancho_logo = 92
        elif "nombre_neurox" in nombre_logo:
            ancho_logo = 72
        else:
            ancho_logo = 30
        pdf.add_centered_image(
            logo_path,
            width_mm=ancho_logo,
            needed_height_mm=28,
            gap_after_image=0.8,
            gap_no_caption=0.8,
        )

    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(33, 63, 104)
    pdf.cell(0, 8, "INFORME DESCRIPTIVO DE ANÁLISIS qEEG", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10.3)
    pdf.set_text_color(220, 122, 40)
    pdf.cell(0, 5.4, "Informe generado automáticamente por NeuroX", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(28, 32, 38)
    pdf.cell(0, 5.5, f"Registro analizado: {registro_titulo}", ln=True, align="C")
    pdf.cell(0, 5.2, f"Fecha de análisis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 5.2, "Software: NeuroX", ln=True, align="C")
    pdf.ln(4.8)

    pdf.add_section_title("Información general")
    pdf.add_visual_table(
        headers=["Parámetro", "Valor"],
        rows=[
            ["Registro analizado", registro_titulo],
            ["Duración", f"{duracion_s:.2f} s"],
            ["Frecuencia de muestreo", f"{fs_meta} Hz"],
            ["Número de canales", str(n_canales_meta)],
            ["Muestras", str(muestras if muestras is not None else "--")],
        ],
        col_widths=[60, 62],
        alignments=["L", "C"],
    )
    pdf.ln(1.0)

    pdf.add_section_title("Alcance del informe")
    pdf.add_wrapped_text(
        "Este informe presenta métricas descriptivas derivadas del procesamiento qEEG. "
        "Los valores reportados son orientativos y no constituyen diagnóstico clínico. "
        "La interpretación debe complementarse con inspección visual de la señal, antecedentes clínicos "
        "y criterio profesional."
    )
    pdf.ln(1.0)

    pdf.add_section_title("Calidad técnica y canales para revisión")
    info_calidad = _texto_calidad_tecnica_compacto_informe(resumen_calidad)
    if isinstance(info_calidad, dict):
        pdf.add_visual_table(
            headers=["Métrica", "Valor"],
            rows=info_calidad["rows"],
            col_widths=[68, 54],
            alignments=["L", "C"],
        )
        if pdf.get_y() > (pdf.page_break_trigger - 70):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 10.8)
            pdf.set_text_color(33, 63, 104)
            pdf.cell(0, 5.5, "Visualización de calidad técnica", ln=True, align="C")
            pdf.set_text_color(28, 32, 38)
            pdf.ln(1.4)
        pdf.add_two_images_row(
            ruta_grafico_calidad,
            ruta_mapa_calidad,
            w_left=62,
            w_right=74,
            caption_left="Figura. Distribución global de calidad técnica por canal.",
            caption_right="Figura. Mapa 10-10 de calidad técnica.",
            gap=12,
            needed_height_mm=78,
        )
        pdf.add_wrapped_text(info_calidad["texto_canales"], font_size=10.2)
    else:
        pdf.add_wrapped_text(str(info_calidad), font_size=10.2)

    pdf.add_section_title("Potencia relativa por bandas")
    pdf.add_wrapped_text(
        "La potencia relativa expresa qué porcentaje de la energía espectral total del EEG se concentra en cada banda de frecuencia, permitiendo comparar la distribución de Delta, Theta, Alfa y Beta entre canales y regiones.",
        font_size=10.2,
    )
    promedios_globales_welch = resumen_welch.get("promedios_globales", {}) if isinstance(resumen_welch, dict) else {}

    pdf.add_block_title("1. Tabla de promedios globales de potencia relativa por bandas")
    pdf.add_visual_table(
        headers=["Banda", "Potencia relativa global"],
        rows=[
            [banda, _fmt_pct_informe(promedios_globales_welch.get(banda, np.nan))]
            for banda in ("Delta", "Theta", "Alfa", "Beta")
        ],
        col_widths=[52, 66],
        alignments=["L", "C"],
    )

    # 2. Tabla regional: se reserva espacio antes del título
    # para evitar que el título quede en una página y la tabla en otra.
    rows_regionales_potencia = _tabla_regional_potencia_informe(resumen_welch)

    alto_tabla_regional = 6.6 * (len(rows_regionales_potencia) + 1)  # filas + encabezado
    alto_titulo_regional = 8.0
    margen_seguridad = 6.0

    pdf.ensure_space(alto_titulo_regional + alto_tabla_regional + margen_seguridad)

    pdf.add_block_title("2. Tabla de promedios regionales de potencia relativa")
    pdf.add_visual_table(
        headers=["Región", "Delta", "Theta", "Alfa", "Beta"],
        rows=rows_regionales_potencia,
        col_widths=[40, 24, 24, 24, 24],
        alignments=["L", "C", "C", "C", "C"],
    )

    pdf.add_block_title("3. Mapa topográfico de energía relativa por bandas")
    if ok_topomaps_welch and os.path.exists(ruta_topomaps_welch):
        pdf.add_centered_image(
            ruta_topomaps_welch,
            width_mm=150,
            caption="Figura. Distribución espacial de la potencia relativa en Delta, Theta, Alfa y Beta.",
            needed_height_mm=122,
        )
    else:
        pdf.add_wrapped_text(
            "No se encontró información suficiente para generar los mapas topográficos de potencia relativa.",
            font_size=10.0,
        )

    pdf.add_section_title("Descripción individual de las bandas")
    pdf.add_wrapped_text(_descripcion_individual_bandas_informe(resumen_welch), font_size=10.2)

    pdf.add_section_title("Proporciones de bandas e índice de lentificación")
    pdf.add_wrapped_text(
        "Las proporciones entre bandas resumen relaciones espectrales relevantes. El índice de lentificación compara la actividad lenta (Delta + Theta) frente a la actividad rápida relativa (Alfa + Beta).",
        font_size=10.2,
    )
    promedios_globales_ratios = resumen_ratios.get("promedios_globales", {}) if isinstance(resumen_ratios, dict) else {}

    pdf.add_block_title("1. Tabla de valores globales de proporciones entre bandas")
    pdf.add_visual_table(
        headers=["Proporción", "Valor global"],
        rows=[
            [RATIOS_WELCH_LABELS[clave], _fmt_num_informe(promedios_globales_ratios.get(clave, np.nan), decimales=2)]
            for clave in ("Theta_Alfa", "Delta_Alfa", "Theta_Beta", "Lentificacion")
        ],
        col_widths=[58, 58],
        alignments=["L", "C"],
    )

    pdf.add_block_title("2. Mapas topográficos de proporciones e índice de lentificación")
    if ok_topomaps_ratios and os.path.exists(ruta_topomaps_ratios):
        pdf.add_centered_image(
            ruta_topomaps_ratios,
            width_mm=150,
            caption="Figura. Distribución espacial de Theta/Alfa, Delta/Alfa, Theta/Beta e índice de lentificación.",
            needed_height_mm=122,
        )
    else:
        pdf.add_wrapped_text(
            "No se encontró información suficiente para generar los mapas topográficos de proporciones de bandas.",
            font_size=10.0,
        )

    pdf.add_section_title("Descripción individual de las proporciones")
    pdf.add_wrapped_text(_descripcion_individual_ratios_informe(resumen_ratios), font_size=10.2)

    pdf.add_section_title("Resumen descriptivo")
    pdf.add_wrapped_text(
        construir_resumen_descriptivo_final_informe(resumen_welch, resumen_ratios),
        font_size=10.4,
    )

    ruta_pdf = os.path.join(carpeta_informe, f"Informe_{nombre_limpio}.pdf")
    pdf.output(ruta_pdf)

    if not os.path.exists(ruta_pdf):
        raise RuntimeError(f"Se intentó generar el PDF pero no apareció: {ruta_pdf}")

    del datos_ica, datos_wavelet, datos_fil, tiempo
    gc.collect()

    ruta_pdf_final = copiar_informe_a_carpeta_externa(
        ruta_pdf,
        carpeta_base_origen,
        logger=logger
    )
    log(f"PDF generado: {ruta_pdf_final}")
    return ruta_pdf_final


# =========================
# PROCESO PRINCIPAL
# =========================
def procesar_archivo(nombre_archivo, carpeta_base=None, logger=None, progress_callback=None):
    """
    Función principal que usará Tkinter.

    Retorna:
        carpeta_archivo (ruta del cache del archivo procesado)
    """
    log = _mklogger(logger)
    silent_logger = lambda _msg: None

    def progress(valor, mensaje=None):
        if callable(progress_callback):
            try:
                progress_callback(float(valor), mensaje)
            except Exception:
                pass

    if carpeta_base is None:
        carpeta_base = CARPETA_BASE

    carpeta_base = os.path.abspath(carpeta_base)
    ruta = os.path.join(carpeta_base, nombre_archivo)

    if not os.path.exists(ruta):
        log(f"[Error] No se pudo cargar el archivo: {nombre_archivo}")
        return None

    rutas = construir_rutas_trabajo(nombre_archivo, carpeta_base)
    crear_carpetas_trabajo(rutas)

    def _formatear_frecuencias_meta(frecuencias):
        if not frecuencias:
            return "ninguna"
        salida = []
        for valor in frecuencias:
            try:
                valor = float(valor)
            except Exception:
                continue
            salida.append(f"{valor:.2f}")
        return ",".join(salida) if salida else "ninguna"

    def _formatear_correcciones_canal_meta(correcciones):
        if not correcciones:
            return "ninguna"
        partes = []
        for item in correcciones:
            nombre = str(item.get("canal", "")).strip() or "Canal"
            freqs_txt = _formatear_frecuencias_meta(item.get("frecuencias_hz", []))
            partes.append(f"{nombre}:{freqs_txt}")
        return ";".join(partes) if partes else "ninguna"

    progress(0, "Preparando procesamiento...")
    log("====================================")
    log(f"[Proceso] Iniciando procesamiento de {nombre_archivo}...")

    # =========================
    # 1) CARGA
    # =========================
    progress(10, "Cargando archivo EEG...")
    log("[Proceso] Cargando archivo EEG...")
    datos_eeg, tiempo, fs_real, meta_dap = cargar_datos_eeg(
        ruta,
        fs_fijo=None,
        n_canales_eeg=n_canales,
        logger=silent_logger
    )
    datos_crudo_para_calidad = datos_eeg.copy()
    log("[OK] Archivo cargado correctamente.")

    nombres_canales_archivo = obtener_nombres_canales_64()
    log(f"[OK] Canales EEG cargados desde lista interna: {nombres_canales_archivo[:8]}")
    
    # =========================
    # 2) CRUDO + VENTANA
    # =========================
    guardar_crudo_y_tiempo(datos_eeg, tiempo, fs_real, rutas["crudo"], logger=silent_logger)

    # =========================
    # 3) FFT CRUDA
    # =========================
    analizar_frecuencia_fft(datos_eeg, fs_real, rutas["fft"], sufijo="crudo", logger=silent_logger)

    # =========================
    # 4) FILTRADO
    # =========================
    progress(20, "Aplicando filtros...")
    log("[Proceso] Aplicando filtros...")
    datos_pb = filtrar_banda(datos_eeg, fs_real, lowcut, highcut, order)
    np.save(
        os.path.join(rutas["filtrado"], "eeg_pasabanda.npy"),
        datos_pb.astype(np.float32, copy=False)
    )
    log("[OK] Filtro pasa banda aplicado.")

    datos_fn = np.asarray(datos_pb, dtype=np.float64)
    if 60.0 < (float(fs_real) / 2.0):
        datos_fn = filtro_notch(datos_fn, fs_real, 60, Q=30)
        log("[OK] Filtro notch aplicado: 60 Hz.")
    else:
        log("[Aviso] No se aplico notch 60 Hz por frecuencia de muestreo insuficiente.")

    resultado_picos = {
        "frecuencias_aplicar": [],
        "frecuencias_candidatas": [],
        "detalles": [],
        "rango_busqueda_hz": [float(AUTO_PICOS_RANGO_BUSQUEDA[0]), float(AUTO_PICOS_RANGO_BUSQUEDA[1])],
        "factor_atenuacion": float(AUTO_PICOS_FACTOR_ATENUACION),
        "criterio": "pico estrecho en proporcion suficiente de canales",
        "max_frecuencias": int(AUTO_PICOS_MAX_FRECUENCIAS),
    }
    frecuencias_globales = []
    if AUTO_DETECTAR_PICOS_TECNICOS:
        resultado_picos = detectar_picos_tecnicos_estrechos(
            datos_fn,
            fs_real,
            logger=logger
        )
        frecuencias_globales = [
            float(f0)
            for f0 in resultado_picos.get("frecuencias_aplicar", [])
            if 0.0 < float(f0) < (float(fs_real) / 2.0)
        ]
        if frecuencias_globales:
            for f0 in frecuencias_globales:
                datos_fn = atenuar_senoidal_exacta(
                    datos_fn,
                    fs_real,
                    freq=float(f0),
                    factor=AUTO_PICOS_FACTOR_ATENUACION,
                    logger=silent_logger
                )
                log(
                    f"[OK] Atenuacion sinusoidal exacta global aplicada: "
                    f"{float(f0):.2f} Hz | factor={AUTO_PICOS_FACTOR_ATENUACION:.2f}."
                )
        else:
            log("[Aviso] No se detectaron picos tecnicos estrechos para atenuacion automatica.")

    config_filtrado = {
        "pasa_banda_hz": [float(lowcut), float(highcut)],
        "orden_pasabanda": int(order),
        "notch_base_hz": [60.0],
        "auto_detectar_picos_tecnicos": bool(AUTO_DETECTAR_PICOS_TECNICOS),
        "tipo_correccion_picos_tecnicos": "atenuacion_sinusoidal_exacta_global",
        "rango_busqueda_picos_hz": [float(AUTO_PICOS_RANGO_BUSQUEDA[0]), float(AUTO_PICOS_RANGO_BUSQUEDA[1])],
        "frecuencias_candidatas_hz": [float(v) for v in resultado_picos.get("frecuencias_candidatas", [])],
        "frecuencias_globales_atenuadas_hz": [float(v) for v in frecuencias_globales],
        "correccion_por_canal": False,
        "correcciones_por_canal": [],
        "factor_atenuacion": float(AUTO_PICOS_FACTOR_ATENUACION),
        "max_frecuencias_atenuadas": int(AUTO_PICOS_MAX_FRECUENCIAS),
        "criterio_global": "pico estrecho en proporcion suficiente de canales",
        "criterio_local": f"pico_frecuencia > {AUTO_PICOS_FACTOR_LOCAL:.1f} * mediana_local",
        "criterio_delta": f"pico_frecuencia > pico_delta_{bandas['Delta'][0]:.0f}_{bandas['Delta'][1]:.0f}Hz",
        "umbral_prop_canales": float(AUTO_PICOS_UMBRAL_PROP_CANALES),
        "ventana_local_hz": float(AUTO_PICOS_VENTANA_LOCAL_HZ),
        "exclusion_central_hz": float(AUTO_PICOS_EXCLUSION_CENTRAL_HZ),
        "ancho_max_hz": float(AUTO_PICOS_ANCHO_MAX_HZ),
        "detalles_picos": [
            {
                "frecuencia_hz": float(item.get("frecuencia_hz", 0.0)),
                "porcentaje_canales_local": float(item.get("porcentaje_canales_local", 0.0)),
                "porcentaje_canales_delta": float(item.get("porcentaje_canales_delta", 0.0)),
                "criterio_usado": str(item.get("criterio_usado", "")),
            }
            for item in resultado_picos.get("detalles", [])
        ],
        "nota": (
            "La correccion se aplica globalmente en frecuencias puntuales mediante atenuacion "
            "sinusoidal exacta. No se usa notch ni correccion individual por canal."
        ),
    }
    with open(os.path.join(rutas["filtrado"], "config_filtrado.json"), "w", encoding="utf-8") as f:
        json.dump(config_filtrado, f, ensure_ascii=False, indent=2)

    np.save(
        os.path.join(rutas["filtrado"], "eeg_filtrado.npy"),
        np.asarray(datos_fn, dtype=np.float32)
    )
    log("[OK] Señal final filtrada guardada.")

    rutas_fft_filtrado = [
        os.path.join(rutas["fft"], "freqs_filtrado.npy"),
        os.path.join(rutas["fft"], "fft_amp_filtrado.npy"),
        os.path.join(rutas["fft"], "fft_db_filtrado.npy"),
        os.path.join(rutas["fft"], "fft_freqs_filtrado.npy"),
    ]
    for ruta_fft in rutas_fft_filtrado:
        if os.path.exists(ruta_fft):
            try:
                os.remove(ruta_fft)
            except Exception:
                pass

    progress(35, "Calculando FFT...")
    analizar_frecuencia_fft(
        datos_fn,
        fs_real,
        rutas["fft"],
        sufijo="filtrado",
        logger=silent_logger
    )
    config_fft = {
        "fft_calculada_desde": "filtrado/eeg_filtrado.npy",
        "senal_final_incluye_atenuacion_picos_tecnicos": bool(frecuencias_globales),
        "tipo_correccion_picos_tecnicos": "atenuacion_sinusoidal_exacta_global",
        "frecuencias_globales_atenuadas_hz": [float(v) for v in frecuencias_globales],
        "correccion_por_canal": False,
        "correcciones_por_canal": [],
        "factor_atenuacion": float(AUTO_PICOS_FACTOR_ATENUACION),
        "fecha_calculo": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(rutas["fft"], "config_fft.json"), "w", encoding="utf-8") as f:
        json.dump(config_fft, f, ensure_ascii=False, indent=2)
    log("[OK] FFT filtrada recalculada desde la señal final.")
    log("[OK] FFT actualizada en cache.")

    progress(45, "Calculando espectrograma...")
    calcular_espectrograma_resumen(
        datos_fn,
        fs_real,
        rutas["espectrograma"],
        sufijo="filtrado",
        logger=silent_logger
    )
    log("[OK] Espectrograma calculado.")

    # Bandas PRE-ICA (desde filtrado)
    for nombre_banda, (low, high) in bandas.items():
        datos_b_pre = filtrar_banda(datos_fn, fs_real, low, high, order)
        np.save(
            os.path.join(rutas["bandas_filtrado"], f"{nombre_banda}.npy"),
            datos_b_pre.astype(np.float32, copy=False)
        )

    # =========================
    # 5) DETECTOR DE CANALES ATÍPICOS
    # =========================
    idx_fuertes, idx_sospechosos, detalle_malos = detectar_canales_atipicos(datos_fn, fs_real)

    guardar_reporte_canales_atipicos(
        rutas["canales_atipicos"],
        idx_fuertes,
        idx_sospechosos,
        detalle_malos,
        nombre_archivo,
        logger=silent_logger
    )

    if len(idx_fuertes) > 0:
        canales_fuertes_txt = ", ".join(
            nombres_canales_archivo[i]
            for i in idx_fuertes
            if 0 <= int(i) < len(nombres_canales_archivo)
        )
        log(f"[Aviso] Canales atipicos fuertes detectados: {canales_fuertes_txt}")
    else:
        log("[OK] No se detectaron canales atipicos fuertes.")

    if len(idx_sospechosos) > 0:
        canales_sosp_txt = ", ".join(
            nombres_canales_archivo[i]
            for i in idx_sospechosos
            if 0 <= int(i) < len(nombres_canales_archivo)
        )
        log(f"[Aviso] Canales sospechosos detectados: {canales_sosp_txt}")
    else:
        log("[OK] No se detectaron canales sospechosos.")

    canales_atipicos_nombres = [
        nombres_canales_archivo[i]
        for i in idx_fuertes
        if 0 <= int(i) < len(nombres_canales_archivo)
    ]
    canales_sospechosos_nombres = [
        nombres_canales_archivo[i]
        for i in idx_sospechosos
        if 0 <= int(i) < len(nombres_canales_archivo)
    ]

    # =========================
    # 6) ICA
    # =========================
    log("[Proceso] Ejecutando ICA y wavelet...")
    datos_ica = aplicar_ica_por_ventanas(datos_fn, fs_real, ventana_seg=10)
    np.save(
        os.path.join(rutas["ica"], "eeg_ica.npy"),
        datos_ica.astype(np.float32, copy=False)
    )

    # =========================
    # 7) WAVELET POST-ICA
    # =========================
    datos_wavelet = aplicar_wavelet_por_canales(datos_ica, logger=silent_logger)
    np.save(
        os.path.join(rutas["wavelet"], "eeg_wavelet.npy"),
        datos_wavelet.astype(np.float32, copy=False)
    )

    # =========================
    # 7b) POTENCIA RELATIVA WELCH
    # =========================
    nombres_canales_welch = nombres_canales_archivo

    progress(55, "Calculando potencia relativa por Welch...")
    resultado_welch = calcular_potencia_relativa_welch(
        datos_wavelet,
        fs_real,
        bandas_def=bandas,
        rango_total=RANGO_TOTAL_WELCH,
        logger=silent_logger
    )
    guardar_potencia_relativa_welch(
        rutas["analisis_welch"],
        resultado_welch,
        nombres_canales=nombres_canales_welch,
        bandas_def=bandas,
        idx_excluir=[],
        logger=silent_logger
    )

    progress(65, "Calculando proporciones de bandas...")
    resultado_ratios_welch = calcular_ratios_welch_principales(
        resultado_welch["pot_rel"],
        ratios_def=RATIOS_WELCH_PRINCIPALES
    )
    guardar_ratios_welch_principales(
        rutas["analisis_welch"],
        resultado_ratios_welch,
        nombres_canales=nombres_canales_welch,
        ratios_def=RATIOS_WELCH_PRINCIPALES,
        idx_excluir=[],
        logger=silent_logger
    )

    log("[OK] Analisis Welch calculado.")

    progress(75, "Evaluando calidad técnica de la señal...")
    calcular_resumen_calidad_senal(
        datos_para_snr=datos_crudo_para_calidad,
        fs=fs_real,
        nombres_canales=nombres_canales_archivo,
        canales_atipicos=canales_atipicos_nombres,
        canales_sospechosos=canales_sospechosos_nombres,
        carpeta_cache=rutas["archivo"],
        logger=logger
    )
    log("[OK] Calidad tecnica de senal calculada.")

    # =========================
    # 8) BANDAS FINALES DESDE WAVELET
    # =========================
    for nombre_banda, (low, high) in bandas.items():
        datos_b_post = filtrar_banda(datos_wavelet, fs_real, low, high, order)
        np.save(
            os.path.join(rutas["bandas"], f"{nombre_banda}.npy"),
            datos_b_post.astype(np.float32, copy=False)
        )

    # =========================
    # 9) META
    # =========================
    progress(85, "Guardando resultados...")
    with open(os.path.join(rutas["archivo"], "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"archivo={nombre_archivo}\n")
        f.write(f"carpeta_base={carpeta_base}\n")
        f.write(f"fs={fs_real}\n")
        f.write(f"data_unit={meta_dap.get('data_unit', 'uV')}\n")
        f.write(f"n_canales={n_canales}\n")
        f.write(f"muestras={datos_eeg.shape[1]}\n")
        f.write(f"duracion_s={datos_eeg.shape[1] / fs_real:.3f}\n")
        f.write("notch_base_hz=60\n")
        f.write(f"auto_detectar_picos_tecnicos={AUTO_DETECTAR_PICOS_TECNICOS}\n")
        f.write("tipo_correccion_picos_tecnicos=atenuacion_sinusoidal_exacta_global\n")
        f.write(
            f"rango_busqueda_picos_hz="
            f"{AUTO_PICOS_RANGO_BUSQUEDA[0]:.0f}-{AUTO_PICOS_RANGO_BUSQUEDA[1]:.0f}\n"
        )
        f.write(f"frecuencias_globales_atenuadas_hz={_formatear_frecuencias_meta(frecuencias_globales)}\n")
        f.write(f"factor_atenuacion_picos={AUTO_PICOS_FACTOR_ATENUACION:.2f}\n")
        f.write("correccion_por_canal=False\n")
        f.write(f"max_frecuencias_atenuadas={AUTO_PICOS_MAX_FRECUENCIAS}\n")
        f.write(
            f"criterio_global="
            f"pico_estrecho_en_mayor_o_igual_a_{AUTO_PICOS_UMBRAL_PROP_CANALES * 100:.0f}pct_de_canales\n"
        )
        f.write(
            f"criterio_delta=pico_frecuencia_mayor_que_pico_delta_"
            f"{bandas['Delta'][0]:.0f}_{bandas['Delta'][1]:.0f}Hz\n"
        )
        f.write(f"criterio_local=pico_frecuencia_mayor_que_{AUTO_PICOS_FACTOR_LOCAL:.1f}x_mediana_local\n")
        f.write(f"umbral_prop_canales={AUTO_PICOS_UMBRAL_PROP_CANALES:.2f}\n")
        f.write(f"ancho_max_picos_hz={AUTO_PICOS_ANCHO_MAX_HZ:.1f}\n")
        f.write("correcciones_por_canal=ninguna\n")

    # Liberar memoria
    del datos_eeg, tiempo, datos_pb, datos_fn, datos_ica, datos_wavelet
    gc.collect()

    log("[OK] Procesamiento completado correctamente.")
    return rutas["archivo"]


# =========================
# EJECUCIÓN MANUAL
# =========================
if __name__ == "__main__":
    carpeta_cache = procesar_archivo(NOMBRE_DAT, carpeta_base=CARPETA_BASE)

    if carpeta_cache:
        ruta_pdf = generar_informe_desde_cache(carpeta_cache)
        print("PDF:", ruta_pdf)

    print("\nPIPELINE + INFORME TERMINADO.")
