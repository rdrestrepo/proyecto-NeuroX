# MEMORY.md — NeuroX

> Documento de **estado vigente, decisiones de arquitectura y reglas técnicas**. No es un changelog: las tareas cerradas no se documentan en detalle. Si algo aquí contradice al código, **gana el código** y este archivo se corrige.

---

## 1. Estado Actual (2026-10-23)

- **Repositorio:** raíz del proyecto y `proyecto-NeuroX/` sincronizadas; `main == origin/main` (último cambio funcional: `43e4c34`, T-4/T-6). Commits firmados como `Kypez-dev <322148154+Kypez-dev@users.noreply.github.com>` (`user.useConfigOnly = true`). Historial previo a la reescritura respaldado en `backup/pre-rewrite-20261019`.
- **Datos fuera de Git:** `Ejecutar/`, `CACHE/` y formatos clínicos (`*.dat`, `*.dap`, `*.edf`, `*.bdf`, `*.ceo`, `*.rs3`, `*.log`) en `.gitignore`. La suite `tests/` también es local (fuera de Git).
- **Cachés:** 143 carpetas en `Ejecutar/CACHE/`, generadas antes de T-4/T-6. Conservan sus métricas históricas hasta reprocesarlas.
- **Suite de pruebas:** **154 pruebas** en 5 archivos (`test_adquisicion.py`, `test_config_adquisicion.py`, `test_cohorte_fz_f2.py`, `test_snr_confiabilidad.py`, `test_regresion_e2e.py`). Con los 5 registros reales: 154/154 OK (~2 min).

### 1.1 Acondicionamiento y filtrado (`pipeline_eeg_completo.py`)

- Hilos BLAS/OpenMP fijados a `"1"` **antes** de importar NumPy/SciPy (`OMP`, `OPENBLAS`, `MKL`, `VECLIB`, `NUMEXPR`).
- Lectura binaria validada contra el `.dap` con `np.memmap`; solo los 64 canales EEG se cargan en RAM como `float32`.
- `detrend` lineal + Butterworth SOS de fase cero (`sosfiltfilt`, 1.0–40.0 Hz, orden 4, coeficientes en `@lru_cache`) + Notch de fase cero a 60 Hz (Q=30).
- `AUTO_DETECTAR_PICOS_TECNICOS = False` por defecto: suprimir picos estrechos amputaría ritmos Alfa/Mu fisiológicos.

### 1.2 Detección de anomalías, interpolación y referencia

- `detectar_canales_atipicos()`: clasifica `fuertes` y `sospechosos` con Z-scores robustos (Mediana/MAD) temporales y espectrales. **Nunca** marca `fuerte` solo por amplitud: exige además rareza espectral.
- `interpolar_canales_malos()` (IDW al cuadrado): por defecto interpola **solo `fuertes`**. **Excepción adaptativa:** si `len(fuertes) / n_canales > 0.10`, también interpola los `sospechosos`.
- `aplicar_car_robusto()`: promedio común con canales corticales sanos, excluyendo `fuertes`, `sospechosos` y periféricos (`M1`, `M2`, `CB1`, `CB2`). Se resta a los 64 canales.

### 1.3 Limpieza de artefactos (ICA + Wavelet)

- **Rama de entrenamiento `datos_ica_fit` (1–100 Hz):** comparte con la señal clínica (1–40 Hz) el `detrend`, la misma interpolación IDW y el **mismo `aplicar_car_robusto()`**. No se aplica un segundo FIR ni `set_eeg_reference('average')`.
- **SNR pre-CAR:** el SNR de adquisición se mide sobre `datos_ica_fit` **antes** del CAR (`snr_crudo_db`) y define el umbral adaptativo de ICLabel:
  - `SNR < 5 dB` → **0.50** (incluye `'other'`, modo agresivo);
  - `SNR < 10 dB` → **0.60** (incluye `'other'`, modo moderado);
  - `SNR >= 10 dB` → **0.70** (modo estándar).
- **Épocas contaminadas:** `_detectar_epocas_contaminadas()` (Z-MAD > 4.5 en ≥2 canales, ventanas de 2 s) excluye saltos **solo para `ica.fit()`**. `ica.apply()` se proyecta sobre la señal continua completa (sin recortes ni saltos de fase).
- **Reintento PCA:** si un canal acapara >99 % de la varianza y `n_components=0.99` deja 1 componente, reintenta con `min(n_eeg − 1, 30)`.
- **Wavelet:** `db4` nivel 4 por canal, umbral universal suave, aplicado **solo a los 2 detalles más finos** (`WAVELET_DETALLES_LIMPIAR = 2`, >62.5 Hz). Las bandas clínicas (1–30 Hz) quedan intactas.

### 1.4 Análisis espectral, biomarcadores y paradigma EO

- Welch PSD 1.0–30.0 Hz, ventana Hann 2.0 s, 50 % solapamiento, `detrend="linear"`, agregación por **mediana** (`WELCH_AVERAGE = "median"`).
- Bandas IFCN: `Delta 1–4`, `Theta 4–8`, `Alfa 8–13`, `Beta 13–30` Hz.
- Exclusión de periféricos centralizada en `indices_canales_corticales()` y `_region_de_canal()`: aplica a promedios globales/regionales, PAF y espectrograma promedio del PDF.
- Condición activa por defecto `CONDICION_REGISTRO = "EO"` (reposo vigil, ojos abiertos con fijación, adultos 20–60 años). Umbrales en `REFERENCIA_BANDAS_EO`, `REFERENCIA_RATIOS_EO`, `GRADIENTE_AP_UMBRALES_EO`; piso `0.00` en ratios de lentificación para evitar falsos "Patológico" en valores bajos.
- **PAF (`calcular_paf_por_canal`):** descuenta la recta secante 1/f entre 8–13 Hz y exige cresta real con `find_peaks` (`prominencia_factor = 0.12`, `ratio_min = 1.18`). Sin cresta por bloqueo visual de Berger → `NaN` → `ESTADO_PAF_SIN_PICO = "Bloqueo visual (Normal EO)"` (verde) en PDF, GUI y Excel; **nunca** "Lentificado".
- **Sesgo de Jensen:** se reporta la media de cocientes por canal (contra la que están calibrados los umbrales EO) y se persiste `cociente_medias` (E[num]/E[den]) en `ratios_promedios_principales.json` para auditoría. Spearman >0.95; **decisión: no cambiar el estimador en producción** mientras no exista cohorte de controles sanos.

### 1.5 Calidad técnica, Confiabilidad y adquisición

- **Fuente única de Confiabilidad:** `calcular_resumen_calidad_senal()` evalúa `_calcular_confiabilidad_metrica()` una vez y guarda `confiabilidad_biomarcadores` en `resumen_calidad.json`. PDF y `consolidado_excel.py` la **leen**; el respaldo para cachés antiguas usa SNR crudo.
- **SNR de la Confiabilidad (T-4):** por defecto (`usar_snr_sin_red_en_confiabilidad = true`) usa `snr_sin_red_db` (crudo 1–30 vs 40–100 Hz descontando 50/60 Hz), coherente con los biomarcadores calculados sobre la señal con notch. Se guarda `confiabilidad_snr` (`fuente`, `snr_db`, `incluye_ruido_red`) para auditoría.
- La varianza ICA removida **no suma puntos** (un sujeto quieto tiene poca varianza removida legítimamente); solo penaliza si es 0 % con SNR <5 dB.
- `diagnosticar_adquisicion()`: traduce fallas a acciones, calcula electrodos para MEDIA/ALTA, evalúa red (≥30 % del ruido 40–100 Hz) y patrón sistémico (≥20 % de canales con la misma causa), y guarda el snapshot `config_adquisicion` en `resumen_calidad.json`.
- **Cohorte FZ/F2:** historial atómico e idempotente en `<carpeta_datos>/cohorte_fz_f2.json`, reconstruible desde `CACHE/`. Alerta si **ambos** fallan como atípico fuerte en ≥80 % de al menos 20 registros. Cohorte actual: FZ 89 %, F2 91 %.
- `generar_informe_desde_cache()` es estrictamente de **solo lectura**: no recalcula ni sobrescribe matrices; genera el PDF en ~2.6 s.

### 1.6 Entregables e interfaz

- **PDF (`fpdf2`):** informe clínico con topomaps interpolados (spline cúbico/gaussiano), sección "Calidad técnica", "Recomendaciones de adquisición" y "Configuración de adquisición aplicada".
- **Excel (`openpyxl`):** consolidado multicapa, `Tabla1` en `A2:AC`, color de `Confiabilidad` en columna `AB`, `Observaciones` en `AC`, banners alineados y hojas ocultas `Detalle1–3` preservadas al final.
- **GUI (`Interfaz_neuroX.py`):** 3 columnas — Registros (izquierda), Visor/Análisis con barra única `tabbar_principal` de **9 pestañas** (6 de señal + 3 de análisis, centro) e Inspector clínico (derecha: semáforos EO, barra apilada de bandas, calidad y recomendaciones de próxima toma).
  - Existe una `tabbar_global` legado `["Gráficas", "Análisis"]` que se conserva por compatibilidad de llamadas, pero **no se muestra**.
- **Consolidado PDF (`pypdf`):** fusión de informes individuales.

---

## 2. Reglas de Arquitectura y Trampas (NO repetir)

1. **Hilos BLAS antes de NumPy/SciPy.** Mantener `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS = "1"` en la cabecera de `pipeline_eeg_completo.py`; de lo contrario `ThreadPoolExecutor` + BLAS multihilo satura la CPU y congela la UI.
2. **`np.memmap` en Windows.** Nunca sobrescribir un `.npy` mapeado. Usar `_np_save_seguro()` en el pipeline y `_liberar_cache_visual()` + `gc.collect()` en la GUI antes de reprocesar. `generar_informe_desde_cache()` jamás debe escribir `.npy`.
3. **Periféricos antes de prefijos.** En cualquier regionalización (`_region_de_canal`, `_region_de_canal_ui`), resolver explícitamente `M1, M2, A1, A2` y `CB1, CB2` **antes** de evaluar `.startswith(("O", "PO", "C"))`. Si no, `CB1/CB2` caen en Occipital/Central y contaminan regiones corticales con señal que ICA no limpia.
4. **Alineación espacial ICA.** `datos_ica_fit` (1–100 Hz) debe llevar exactamente el mismo `aplicar_car_robusto()` que `datos_fn` (1–40 Hz). No aplicar un segundo FIR ni `set_eeg_reference('average')` dentro de `aplicar_ica_mne()`.
5. **SNR de adquisición antes del CAR.** El CAR altera el SNR crudo (puede pasar de −2 a +18 dB) y engaña al umbral adaptativo de ICLabel. Medir `snr_crudo_db` antes de `aplicar_car_robusto()`.
6. **No recortar épocas en la señal de salida.** `_detectar_epocas_contaminadas()` solo recorta muestras para `ica.fit()`. `ica.apply()` se proyecta sobre la matriz continua completa para no crear discontinuidades ni energía falsa en Delta.
7. **Compatibilidad NumPy 2.x y `fpdf2`.** Usar siempre `_trapz()` (`np.trapezoid` / `np.trapz`) y `_normalizar_imagen_para_pdf()` (PIL) antes de entregar imágenes a `fpdf2`.
8. **Interpolación vs. exclusión.** Interpolar `sospechosos` solo cuando `ratio_atipicos > 0.10`; en el CAR y en los promedios de Welch se excluyen tanto `fuertes` como `sospechosos`. Nunca interpolar un sospechoso de forma indiscriminada: falsifica la varianza cortical.
9. **Tres SNRs distintos, no mezclar:**
   - `snr_global_db`: crudo, 1–30 vs 40–100 Hz, señal original (calidad de toma);
   - `snr_sin_red_db`: igual pero descontando 50/60 Hz ±2 Hz (insumo por defecto de la Confiabilidad);
   - `snr_final_db`: post-limpieza, 1–30 vs 30–40 Hz sobre `eeg_wavelet.npy`.
10. **Idioma y terminología.** Código y variables en español; informes en español formal. **Nunca** afirmar "diagnóstico de DCL": usar "hallazgo descriptivo", "indicadores compatibles", "patrón de lentificación".

---

## 3. Próximos Pasos (pendientes reales)

### A. Código y empaquetado

- [ ] **Reprocesamiento masivo:** las 143 cachés de `Ejecutar/CACHE/` y sus informes preceden a T-4/T-6. Respaldar/renombrar `CACHE/` y reprocesar el lote (~15–25 s/archivo) + regenerar PDFs y consolidado.
- [ ] **Validar el `.exe` (`NeuroX.spec`):** compilar y verificar en entorno limpio `assets/`, `config.json` y el modelo ONNX de `mne_icalabel`.
- [ ] **Bordes de banda en `_integrar_psd_por_banda()`:** usa `freqs < fmax`, lo que descarta el último trapecio de cada banda (3.5–4.0, 7.5–8.0, 12.5–13.0 Hz). Cambiar a `<=` para que las 4 bandas sumen 100.0 % del total. **Cambio intencional:** exige regenerar referencias E2E.
- [ ] **Región sin canales sanos:** en `energia_relativa_por_region()`, si todos los canales de una región quedan excluidos, la región desaparece del informe/Excel. Añadir respaldo con canales interpolados.
- [ ] **Asimetría interhemisférica frontal** (opcional, investigación): biomarcador complementario.

### B. Operación y hardware (fuera del software)

- [ ] **Reparar FZ/F2:** continuidad de electrodos, cables y pines del conector (o cambiar gorro). Fallan como atípico fuerte en ~90 % de la cohorte.
- [ ] **Mitigar red de 60 Hz:** unificar GND paciente/amplificador, igualar impedancias REF/GND y separar cables de alimentación antes de grabar.
- [ ] **Depurar archivos de origen:** re-adquirir o descartar `DCL 026` y `DCL 084` (22 canales), `DCL 042` (`.dap` corrupto, `NumSamples = −1`) y `DCL 119` (grabación de 0.4 s).

---

## 4. Comandos Operativos (PowerShell, desde la raíz)

```powershell
# 1. Dependencias
pip install -r requirements.txt

# 2. GUI
python Interfaz_neuroX.py

# 3. Motor headless (archivo de prueba configurado en el módulo)
python pipeline_eeg_completo.py

# 4. Verificación sintáctica (pasada única)
python -m py_compile Interfaz_neuroX.py pipeline_eeg_completo.py consolidado_excel.py consolidado_pdf.py
if ($LASTEXITCODE -eq 0) { "VERIFICACION_SINTACTICA_OK" }

# 5. Suite completa rápida (~20 s; 1 prueba real omitida)
python -m unittest discover -s tests -p "test_*.py" --quiet

# 6. Suite con los 5 registros reales (~2 min)
$env:NEUROX_DATOS_REGRESION = (Resolve-Path 'Ejecutar').Path
python -m unittest discover -s tests -p "test_*.py" --quiet

# 7. Omitir la regresión E2E pesada
$env:NEUROX_SALTAR_E2E = "1"

# 8. Regenerar referencias SOLO tras un cambio intencional de resultados
python tests/regresion_e2e.py --actualizar --reales "Ejecutar"

# 9. Compilar ejecutable
pyinstaller NeuroX.spec
```

---

## 5. Estructura de Caché (por registro)

```
CACHE/<registro_limpio>/
├── crudo/              # eeg_crudo.npy, tiempo.npy
├── fft/                # Espectros unilaterales en µV y dB: crudo, filtrado (post-CAR) y final
├── espectrograma/      # STFT resumida: freqs, times y JSON por canal
├── filtrado/           # eeg_filtrado.npy (detrend + pasa-banda + notch + IDW + CAR)
├── canales_atipicos/   # Índices/métricas Z-MAD y reporte_canales_atipicos.txt
├── ica/                # eeg_ica.npy, calidad_ica.json (% varianza removida)
├── wavelet/            # eeg_wavelet.npy (señal final)
├── analisis_welch/     # PSD, potencias abs/rel, PAF, Gradiente AP, ratios (.npy/.npz/.json)
├── calidad_senal/      # resumen_calidad.json (SNR, confiabilidad, config_adquisicion, recomendaciones)
├── bandas/             # Delta/Theta/Alfa/Beta temporales (.npy)
├── informe/            # Informe_<registro>.pdf y figuras intermedias
└── meta.txt            # fs, duración, canales, configuración aplicada
```

Fuera de `CACHE/`: `<carpeta_datos>/cohorte_fz_f2.json` (historial de hardware FZ/F2).

> **Cachés previas a T-4/T-6** no tienen `config_adquisicion` ni `confiabilidad_snr`: el PDF/inspector muestran la configuración vigente con aviso explícito y el respaldo de Confiabilidad usa el SNR crudo histórico (paridad PDF/Excel).

---

## 6. Configuración (`config.json`)

```json
{
  "adquisicion": {
    "umbral_interferencia_red_pct": 30.0,
    "umbral_patron_sistemico": 0.20,
    "umbral_patron_cohorte_fz_f2": 0.80,
    "min_registros_cohorte": 20,
    "usar_snr_sin_red_en_confiabilidad": true
  }
}
```

- Claves que empiezan por `_` son documentación y se ignoran.
- Loader tolerante: falta/corrupción/valores inválidos → defaults + avisos en `CONFIG_ADQUISICION_ADVERTENCIAS` (visibles en PDF, inspector y log).
- Booleano estricto: solo `true`/`false`; `1`, `"true"`, `null` se rechazan.
- Enteros validados: `20.5` se rechaza y cae al default.
- En `.exe`: busca junto al ejecutable y luego en `_internal` (`sys._MEIPASS`).

---

## 7. Referencias Clave en Código

| Función / Constante | Archivo | Propósito |
|---------------------|---------|-----------|
| `calcular_resumen_calidad_senal()` | `pipeline_eeg_completo.py` | Única fuente de `confiabilidad_biomarcadores`; PDF y Excel la leen. |
| `_snr_para_confiabilidad()` | `pipeline_eeg_completo.py` | Devuelve `(snr_db, fuente)` según el flag T-4. |
| `diagnosticar_adquisicion(..., usar_snr_sin_red=None)` | `pipeline_eeg_completo.py` | Usa la misma fuente que la Confiabilidad; el parámetro fuerza el criterio al reconstruir cachés. |
| `config_adquisicion_desde_cache()` | `pipeline_eeg_completo.py` | Snapshot trazable para PDF/inspector; nunca lanza. |
| `clasificar_semaforo()` | `pipeline_eeg_completo.py` | Única fuente de verdad de umbrales clínicos (pipeline + GUI). |
| `indices_canales_corticales()` | `pipeline_eeg_completo.py` | Exclusión centralizada de periféricos. |
| `_calcular_confiabilidad_metrica()` | `pipeline_eeg_completo.py` | SNR + % atípicos + % varianza ICA → NULA/BAJA/MEDIA/ALTA. |
| `_integrar_psd_por_banda()` | `pipeline_eeg_completo.py` | Integración trapezoidal por banda (ver pendiente de bordes). |
| `_trapz()` | `pipeline_eeg_completo.py` | Polifill NumPy 1.x/2.x. |
| `_np_save_seguro()` | `pipeline_eeg_completo.py` | Escritura segura de `.npy` en Windows. |
| `ETIQUETAS_TABS_PRINCIPALES` | `Interfaz_neuroX.py` | 9 pestañas visibles (6 señal + 3 análisis). |
