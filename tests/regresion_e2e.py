# -*- coding: utf-8 -*-
"""
T-5 — Regresión end-to-end del pipeline NeuroX.

Herramientas compartidas por tests/test_regresion_e2e.py:

- generar_registro_sintetico(): crea un par .dat/.dap reproducible (misma
  semilla = mismos bytes) con fondo 1/f, alfa occipital, theta frontal,
  parpadeos, red de 60 Hz y tres canales defectuosos a propósito.
- extraer_huella(): resume una carpeta de CACHE en un dict plano con las
  métricas clínicas y técnicas que no deben cambiar sin intención
  (bandas, ratios, semáforos, PAF, gradiente, calidad, SNR, red, ICA y
  recomendaciones de adquisición).
- comparar_huellas(): diferencias contra la referencia con tolerancias por
  tipo de métrica (las pequeñas variaciones numéricas entre equipos, p. ej.
  BLAS distinto en el ICA, no deben romper la prueba; un cambio de
  categoría o de canales detectados sí).

Regenerar referencias SOLO cuando un cambio de resultados sea intencional:
    python tests/regresion_e2e.py --actualizar
    python tests/regresion_e2e.py --actualizar --reales "C:/ruta/datosTodos"
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

CARPETA_REFERENCIAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "referencias")
REF_SINTETICO = os.path.join(CARPETA_REFERENCIAS, "e2e_sintetico.json")
REF_REALES = os.path.join(CARPETA_REFERENCIAS, "e2e_reales.json")

# Variable de entorno con la carpeta de los .dat/.dap reales (opcional).
ENV_DATOS_REALES = "NEUROX_DATOS_REGRESION"
# Registros reales de la auditoría 2026-10-19 (D-01..M-02).
REGISTROS_REALES = ("DCL 003.dat", "DCL 007.dat", "DCL 024.dat", "DCL 083.dat", "DCL 138.dat")

NOMBRE_SINTETICO = "SINTETICO E2E.dat"
VERSION_HUELLA = 1

# Canales defectuosos del registro sintético (nombre -> defecto).
CANALES_DEFECTUOSOS = {"T7": "ruido_hf", "PO7": "plano", "C4": "amplitud_y_ruido"}

# Tolerancias por prefijo de clave. Las claves sin tolerancia se comparan
# exactas (categorías, listas de canales, conteos).
TOLERANCIAS = (
    ("bandas.global.", ("abs", 0.5)),        # puntos porcentuales
    ("bandas.region.", ("abs", 1.0)),
    ("ratios.global.", ("rel", 0.03)),
    ("ratios.cociente_medias.", ("rel", 0.03)),
    ("paf.", ("abs", 0.15)),                 # Hz
    ("gradiente_ap", ("rel", 0.03)),
    ("alfa.", ("abs", 0.5)),
    ("calidad.snr_", ("abs", 0.1)),          # dB
    ("calidad.pct_ruido_red", ("abs", 0.5)),
    ("ica.porcentaje_varianza_removida", ("abs", 2.0)),
)


# =========================
# Registro sintético
# =========================
def _ruido_rosa(rng, n_fuentes, n_muestras):
    """Ruido 1/f (rosa) por síntesis espectral, varianza unitaria."""
    blanco = rng.standard_normal((n_fuentes, n_muestras))
    espectro = np.fft.rfft(blanco, axis=1)
    f = np.arange(espectro.shape[1], dtype=np.float64)
    f[0] = 1.0
    espectro /= np.sqrt(f)
    x = np.fft.irfft(espectro, n=n_muestras, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True) + 1e-12
    return x


# Posición antero-posterior por prefijo (0 = polo frontal, 4 = occipital).
_FILAS_AP = (("FP", 0.0), ("AF", 0.5), ("FT", 1.5), ("FC", 1.5), ("F", 1.0), ("TP", 2.5),
             ("CP", 2.5), ("T", 2.0), ("C", 2.0), ("M", 2.5), ("PO", 3.5), ("P", 3.0),
             ("CB", 4.0), ("O", 4.0))
GANANCIA_ALFA = 3.0


def _fila_ap(nombres):
    filas = []
    for c in nombres:
        filas.append(next((v for p, v in _FILAS_AP if c.startswith(p)), 2.0))
    return np.array(filas, dtype=np.float64)


def generar_registro_sintetico(carpeta, nombre=NOMBRE_SINTETICO, semilla=7, duracion_s=120.0, fs=500.0):
    """
    Escribe <carpeta>/<nombre>.dat y .dap (FLOAT32, orden SAMP, 64 EEG + 5
    auxiliares, en uV) y devuelve la ruta del .dat. Determinista.
    """
    import pipeline_eeg_completo as P

    rng = np.random.default_rng(semilla)
    nombres = [P.limpiar_nombre_canal(c).upper() for c in P.obtener_nombres_canales_64()]
    n_eeg = len(nombres)
    n_aux = 5
    n = int(round(duracion_s * fs))
    t = np.arange(n, dtype=np.float64) / fs

    # Fondo cortical: 16 fuentes 1/f mezcladas con pesos suaves para que los
    # canales vecinos estén correlacionados (como en un EEG real).
    fuentes = _ruido_rosa(rng, 16, n)
    mezcla = rng.normal(0.0, 1.0, (n_eeg, 16))
    eeg = 8.0 * (mezcla @ fuentes) / np.sqrt(16.0)

    def peso(prefijos, valor):
        return np.array([valor if any(c.startswith(p) for p in prefijos) else 0.0 for c in nombres])

    # Alfa 10 Hz con modulación lenta, dominante en occipital/parietal.
    envolvente = 1.0 + 0.4 * np.sin(2 * np.pi * 0.1 * t)
    alfa = envolvente * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    # Gradiente antero-posterior suave (fisiológico): sin saltos entre filas
    # para que el detector no confunda el alfa normal con un canal raro.
    eeg += np.outer(GANANCIA_ALFA * _fila_ap(nombres), alfa)

    # Theta 6 Hz frontal moderada.
    theta = np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
    eeg += np.outer(peso(("F", "AF", "FC"), 2.0), theta)

    # Parpadeos (gaussianas de ~200 ms cada ~4 s) en polos frontales.
    parpadeo = np.zeros(n)
    for c in np.arange(2.0, duracion_s - 1.0, 4.0):
        centro = c + rng.uniform(-0.5, 0.5)
        parpadeo += 120.0 * np.exp(-0.5 * ((t - centro) / 0.08) ** 2)
    eeg += np.outer(peso(("FP",), 1.0) + peso(("AF",), 0.5) + peso(("F",), 0.15), parpadeo)

    # Red eléctrica de 60 Hz común a todos los canales.
    eeg += 4.0 * np.sin(2 * np.pi * 60.0 * t)[None, :]

    # Canales defectuosos.
    i = nombres.index("T7")
    eeg[i] += 40.0 * rng.standard_normal(n)              # ruido de alta frecuencia
    i = nombres.index("PO7")
    eeg[i] *= 0.02                                       # canal casi plano
    i = nombres.index("C4")
    eeg[i] = 5.0 * eeg[i] + 25.0 * rng.standard_normal(n)  # electrodo suelto: amplitud + ruido

    aux = 20.0 * rng.standard_normal((n_aux, n))
    datos = np.vstack([eeg, aux]).astype(np.float32)

    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, os.path.splitext(nombre)[0])
    datos.T.tofile(base + ".dat")                        # orden SAMP: muestra a muestra
    with open(base + ".dap", "w", encoding="utf-8") as f:
        f.write(
            "DataFormat = FLOAT\n"
            f"NumChannels = {n_eeg + n_aux}\n"
            f"NumSamples = {n}\n"
            f"SampleFreqHz = {fs:.1f}\n"
            "DataUnit = uV\n"
            "DataSampOrder = SAMP\n"
        )
    return base + ".dat"


# =========================
# Huella de una caché
# =========================
def _leer_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def extraer_huella(carpeta_cache):
    """Dict plano y ordenado con las métricas que fija la regresión."""
    h = {"version": VERSION_HUELLA}

    meta = {}
    with open(os.path.join(carpeta_cache, "meta.txt"), "r", encoding="utf-8") as f:
        for linea in f:
            if "=" in linea:
                k, v = linea.rstrip("\n").split("=", 1)
                meta[k] = v
    h["meta.fs"] = _num(meta.get("fs"))
    h["meta.n_canales"] = int(meta.get("n_canales", 0))
    h["meta.muestras"] = int(meta.get("muestras", 0))

    pot = _leer_json(os.path.join(carpeta_cache, "analisis_welch", "pot_rel_promedios.json"))
    for banda, v in sorted(pot["promedios_globales"].items()):
        h[f"bandas.global.{banda}"] = _num(v)
    for region, bandas in sorted(pot["promedios_regionales"].items()):
        for banda, v in sorted(bandas.items()):
            h[f"bandas.region.{region}.{banda}"] = _num(v)
    h["paf.global_hz"] = _num(pot.get("paf_global_hz"))
    h["paf.occipital_hz"] = _num(pot.get("paf_occipital_hz"))
    h["paf_canales_validos"] = pot.get("paf_canales_validos")
    h["gradiente_ap"] = _num(pot.get("gradiente_ap"))
    h["alfa.occipital"] = _num(pot.get("alfa_occipital"))
    h["alfa.frontal"] = _num(pot.get("alfa_frontal"))

    rat = _leer_json(os.path.join(carpeta_cache, "analisis_welch", "ratios_promedios_principales.json"))
    for k, v in sorted(rat["promedios_globales"].items()):
        h[f"ratios.global.{k}"] = _num(v)
    for k, v in sorted((rat.get("cociente_medias") or {}).items()):
        h[f"ratios.cociente_medias.{k}"] = _num(v)
    for k, v in sorted(rat["semaforos"].items()):
        h[f"semaforo.{k}"] = v.get("estado")

    cal = _leer_json(os.path.join(carpeta_cache, "calidad_senal", "resumen_calidad.json"))
    h["calidad.canales_atipicos"] = sorted(cal.get("canales_atipicos") or [])
    h["calidad.canales_sospechosos"] = sorted(cal.get("canales_sospechosos") or [])
    h["calidad.estado_global"] = cal.get("estado_global")
    h["calidad.confiabilidad"] = cal.get("confiabilidad_biomarcadores")
    snr = cal.get("snr") or {}
    h["calidad.snr_global_db"] = _num(snr.get("snr_global_db"))
    h["calidad.snr_sin_red_db"] = _num(snr.get("snr_sin_red_db"))
    h["calidad.pct_ruido_red"] = _num(snr.get("pct_ruido_red"))

    rec = cal.get("recomendaciones_adquisicion") or {}
    pm, pa = rec.get("para_media"), rec.get("para_alta")
    h["adquisicion.para_media"] = pm["electrodos"] if pm else None
    h["adquisicion.para_alta"] = pa["electrodos"] if pa else None
    h["adquisicion.patron_sistemico"] = (rec.get("patron_sistemico") or {}).get("causa")
    ir = rec.get("interferencia_red")
    h["adquisicion.interferencia_red"] = bool(ir)
    h["adquisicion.confiabilidad_sin_red"] = ir.get("confiabilidad_sin_red") if ir else None

    ica = os.path.join(carpeta_cache, "ica", "calidad_ica.json")
    h["ica.porcentaje_varianza_removida"] = _num(_leer_json(ica).get("porcentaje_varianza_removida")) \
        if os.path.exists(ica) else None
    return h


def _tolerancia(clave):
    for prefijo, tol in TOLERANCIAS:
        if clave.startswith(prefijo):
            return tol
    return None


def comparar_huellas(referencia, actual):
    """Lista de textos con cada diferencia fuera de tolerancia (vacía = OK)."""
    difs = []
    for clave in sorted(set(referencia) | set(actual)):
        if clave not in actual:
            difs.append(f"{clave}: falta en el resultado actual (referencia {referencia[clave]!r})")
            continue
        if clave not in referencia:
            difs.append(f"{clave}: clave nueva {actual[clave]!r} (regenerar referencias si es intencional)")
            continue
        ref, act = referencia[clave], actual[clave]
        tol = _tolerancia(clave)
        if tol is None or ref is None or act is None:
            if ref != act:
                difs.append(f"{clave}: referencia {ref!r} -> actual {act!r}")
            continue
        tipo, valor = tol
        delta = abs(float(act) - float(ref))
        limite = valor if tipo == "abs" else valor * max(abs(float(ref)), 1e-9)
        if delta > limite:
            difs.append(f"{clave}: referencia {ref:.4f} -> actual {act:.4f} (dif {delta:.4f} > tol {limite:.4f})")
    return difs


# =========================
# Ejecución del pipeline aislado
# =========================
def procesar_en_temporal(ruta_dat, carpeta_trabajo, con_informe=False):
    """
    Copia el .dat/.dap a carpeta_trabajo y corre procesar_archivo allí, sin
    tocar la CACHE ni los informes del usuario. Devuelve la carpeta de caché
    o, con con_informe=True, (carpeta de caché, ruta del PDF).
    """
    import pipeline_eeg_completo as P

    os.makedirs(carpeta_trabajo, exist_ok=True)
    base = os.path.splitext(ruta_dat)[0]
    nombre = os.path.basename(ruta_dat)
    if os.path.dirname(os.path.abspath(ruta_dat)) != os.path.abspath(carpeta_trabajo):
        for ext in (".dat", ".dap", ".rs3"):
            if os.path.exists(base + ext):
                shutil.copy2(base + ext, os.path.join(carpeta_trabajo, os.path.basename(base) + ext))
    cache = P.procesar_archivo(nombre, carpeta_base=carpeta_trabajo, logger=lambda m: None)
    if not con_informe:
        return cache
    return cache, P.generar_informe_desde_cache(cache, logger=lambda m: None)


def _guardar_json(ruta, datos):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def actualizar_referencias(carpeta_reales=None):
    import warnings
    warnings.filterwarnings("ignore")
    with tempfile.TemporaryDirectory(prefix="neurox_e2e_") as tmp:
        ruta = generar_registro_sintetico(tmp)
        cache = procesar_en_temporal(ruta, tmp)
        _guardar_json(REF_SINTETICO, extraer_huella(cache))
        print(f"[OK] Referencia sintética: {REF_SINTETICO}")

    if carpeta_reales:
        huellas = {}
        for nombre in REGISTROS_REALES:
            origen = os.path.join(carpeta_reales, nombre)
            if not os.path.exists(origen):
                print(f"[Aviso] No existe {origen}; se omite.")
                continue
            with tempfile.TemporaryDirectory(prefix="neurox_e2e_") as tmp:
                cache = procesar_en_temporal(origen, tmp)
                huellas[nombre] = extraer_huella(cache)
            print(f"[OK] {nombre}")
        _guardar_json(REF_REALES, huellas)
        print(f"[OK] Referencias reales: {REF_REALES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regresión end-to-end de NeuroX")
    ap.add_argument("--actualizar", action="store_true", help="Regenerar las referencias")
    ap.add_argument("--reales", default=None, help="Carpeta con los .dat/.dap reales de la auditoría")
    args = ap.parse_args()
    if not args.actualizar:
        ap.error("Use --actualizar para regenerar referencias; para comprobar, ejecute los tests.")
    actualizar_referencias(args.reales)
