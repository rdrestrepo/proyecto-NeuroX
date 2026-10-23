# MEMORY.md — NeuroX

## Estado Actual
- **Sincronización (2026-10-23):** la raíz del proyecto y `proyecto-NeuroX/` son idénticas y `main == origin/main` (tras T-5 y la higiene del repositorio, ver más abajo). Los commits se firman como `Kypez-dev <322148154+Kypez-dev@users.noreply.github.com>`; el historial previo a la reescritura quedó respaldado en la rama `backup/pre-rewrite-20261019`.
- **Pipeline biofísico operativo:** Motor clínico implementado en `pipeline_eeg_completo.py` con acondicionamiento pasa-banda ($1.0\text{–}40\text{ Hz}$), notch a $60\text{ Hz}$ y eliminación de deriva lineal (`detrend`).
- **Limpieza de artefactos:**
  - Detección estadística robusta de canales malos (`detectar_canales_atipicos`) usando Z-score robusto (MAD).
  - Interpolación espacial por distancia inversa (IDW) aplicada exclusivamente a canales con anomalías fuertes.
  - CAR robusto excluyendo canales atípicos, sospechosos y electrodos de mastoides/cerebelo (`M1`, `M2`, `CB1`, `CB2`).
  - ICA clínico basado en Infomax Extendido + red neuronal ICLabel (`mne-icalabel`) para descarte selectivo de componentes oculares y musculares (umbral $\ge 0.70$).
  - Micro-limpieza de ruido de sensor mediante Wavelet Daubechies 4 (`db4`) nivel 4 con soft-thresholding SOLO en los detalles de alta frecuencia (> banda clínica), preservando Alfa/Beta.
- **Análisis espectral e informes:**
  - Welch PSD parametrizado de $1.0\text{ a }30.0\text{ Hz}$, con promedio por MEDIANA de ventanas Hann de 2 s (robusto a parpadeos/movimientos transitorios) y ratios clínicos calculados (Theta/Alfa, Delta/Alfa, Theta/Beta, Índice de Lentificación).
  - Estimación de Frecuencia Pico de Alfa (PAF) con descuento de la pendiente 1/f local + `find_peaks`: distingue un pico real de una simple rampa descendente (evita falsos "Lentificado" en Ojos Abiertos).
  - Generación de informe PDF de grado médico estructurado con `fpdf2` y mapa topográfico 2D interpolado por spline cúbico/gaussiano.
  - Tema visual unificado de gráficos (`aplicar_tema_graficos`, paletas `COLORES_BANDAS`/`COLOR_*`) y guardado seguro de `.npy` (`_np_save_seguro`) para evitar bloqueos de archivo en Windows.
  - Exclusión de periféricos centralizada en `indices_canales_corticales()` y aplicada tanto a los promedios de bandas como a los ratios.
  - Herramientas de exportación masiva: consolidación multicapa en Excel (`openpyxl`) y combinación de PDFs (`pypdf`).
- **Biomarcadores DCL y condición de registro (EO/EC):**
  - Referencias clínicas conmutables por condición vía `CONDICION_REGISTRO` (`"EO"` ojos abiertos con fijación por defecto, `"EC"` ojos cerrados legado).
  - Bandas con zonas asimétricas normal/borderline/patológico (`REFERENCIA_BANDAS_EO`), ratios EO (`REFERENCIA_RATIOS_EO`), Gradiente AP EO (`GRADIENTE_AP_UMBRALES_EO`) y PAF (independiente de condición).
  - `evaluar_banda()` y `clasificar_semaforo()` son la única fuente de verdad de umbrales; la GUI (`_actualizar_biomarcadores`) reutiliza `clasificar_semaforo()` y el Excel replica PAF/Gradiente en `clasificar_semaforo_simple()`.
  - Auditoría del sesgo de Jensen: `cociente_medias` (E[num]/E[den]) se guarda en `ratios_promedios_principales.json` junto a `promedios_globales` (media de cocientes).
- **Interfaz (GUI):**
  - Tkinter/ttk con paleta Índigo/Turquesa, tabs tipo "pill" antialiased por PIL (`_TabBar`) y navegación fluida de ventanas temporales de señal.
  - Rediseño `052f2d7`: barra superior (registro, condición, fs, canales, duración y chip de calidad), columna de Registros a la izquierda, visor central e inspector contextual a la derecha.
  - Una sola fila de pestañas (6 vistas de señal + 3 de análisis) sincronizada con los modos existentes; controles del visor en barra horizontal sobre el gráfico.
  - Procesar / Ver informe PDF / Consolidados bajo la lista de archivos (reducida a 8 filas); cada archivo muestra un punto de estado (procesado/pendiente).
  - Inspector en modo visor: biomarcadores con semáforo, barra apilada de potencia relativa global, calidad técnica y mapa regional.

---

## Auditoría técnica 2026-10-19 (fases 1-4)

Correcciones aplicadas y validadas sobre 5 registros reales (`DCL 003/007/024/083/138`), comparando contra el baseline previo.

- **D-01 — CB1/CB2 contaminaban la región Occipital.** `_region_de_canal()` los clasificaba como `"Occipital"` por `startswith("CB")` y `_region_de_canal_ui()` igual. Ahora devuelven `"Referencia/Periféricos"` antes de evaluar prefijos. `energia_relativa_por_region()` descarta explícitamente `Mastoides`, `Otro` y `Referencia/Periféricos`. Efecto medido: Occipital pasó a ser la media exacta de los canales occipitales corticales (error 0.0 en los 5 registros); en DCL 003 Delta occipital 18.54 % → 16.56 %.
- **A-01 — Welch se llamaba con `idx_excluir=[]`.** Los canales fuertes y sospechosos seguían pesando en promedios globales/regionales aunque estuvieran interpolados o excluidos del CAR. Ahora `procesar_archivo` propaga `idx_malos_totales` a `calcular_potencia_relativa_welch`, `guardar_potencia_relativa_welch`, `calcular_ratios_welch_principales` y `guardar_ratios_welch_principales`. El PAF global también se restringe a canales corticales.
- **D-02 — Confiabilidad distinta entre PDF y Excel y con insumos distintos.** Se unificó en una sola fuente de verdad: `calcular_resumen_calidad_senal()` calcula `confiabilidad_biomarcadores` (SNR efectivo 1-30 vs 40-100 Hz —desde T-4, sin la red de 50/60 Hz por defecto— + % atípicos + % varianza ICA) y la guarda en `resumen_calidad.json`. El PDF y `consolidado_excel.py` la LEEN; solo si falta (caché antigua) recalculan con la misma lógica. La varianza ICA dejó de premiarse (+2 por ≥15 %): un sujeto limpio sin artefactos tiene legítimamente varianza removida baja. Solo penaliza ICA que no removió nada con SNR bajo. Paridad verificada 5/5.
- **D-03 — Columnas desplazadas en Excel.** `Confiabilidad` se coloreaba en `Y` (Sospechosos) y las tablas cubrían `A2:AB` dejando fuera `Observaciones` (AC). Corregido a `AB`, `A2:AC`, `A3:AC`, ancho de `AC` y banner `CALIDAD DE LA SEÑAL` movido de `R1` (encima de PAF) a `X1`, añadiendo `R1 = PAF Y GRADIENTE AP`. **Hallazgo nuevo:** las hojas ocultas `Detalle1-3` se creaban y luego se descartaban al reasignar `wb._sheets`; ahora se conservan al final (Tablas 5/6 restauradas).
- **S-01 — Caché inflada por artefactos de escritura.** Se eliminaron `bandas_filtrado/` (nunca se consumía), los directorios `energia_relativa*` recalculados dentro de `generar_informe_desde_cache()` y las 4 lecturas `np.memmap` que ese método abría. El generador de informes pasó a ser estrictamente de solo lectura. Efecto medido: **-24.7 % de caché en los 5 registros** (~150 MB/registro) y generación de PDF de ~20 s a ~2.6 s.
- **M-01 — ICA entrenado y aplicado sobre referencias distintas.** `datos_ica_fit` se filtraba 1-100 Hz pero no recibía CAR; luego `aplicar_ica_mne()` le aplicaba un FIR 1-100 adicional y `set_eeg_reference('average')` sobre los 60 canales EEG, mientras la señal de trabajo usaba CAR robusto excluyendo sospechosos. Ahora `datos_ica_fit` recibe el mismo `aplicar_car_robusto()` y se eliminaron el segundo filtro y la segunda referencia. El SNR que alimenta el umbral adaptativo de ICLabel se mide ANTES del CAR y se pasa como `snr_crudo_db`, para no alterar el modo adaptativo por el cambio de referencia.
- **M-02 — Espectrograma del informe promediaba periféricos.** `crear_espectrograma_promedio_informe()` ahora acepta `nombres_canales` y promedia solo canales corticales (acumulación por canal para no duplicar el cubo en RAM).
- **D-04 — Control de escala PSD sin efecto.** `frame_psd_opciones` no se mostraba en el modo "FFT filtrada" y `_graficar_fft_filtrada()` forzaba escala lineal. Ahora el check "Eje Y logarítmico" se muestra y alterna entre PSD lineal y dB; las bandas y los cortes se toman de `BANDAS_CLINICAS` (Delta 1-4 Hz, ya no 0.5-4).
- **D-05 — `NeuroX.spec` incompleto.** Se empaqueta la carpeta `assets/` completa, se recolectan los datos de `mne_icalabel` (ICLabelNet.onnx) y los montajes de `mne`, se añade `onnxruntime` a `hiddenimports` y se retira `sklearn.decomposition` (ya no está en `requirements.txt`). Se recuperó `assets/logo_neurox.ico`, que faltaba en el árbol de trabajo.

### Fase 5 — Experimento H-01 (sesgo de Jensen)

Script de solo lectura sobre 156 cachés con `ratios_welch_principales.npz`: comparó la **media de cocientes por canal** (A, la que usa hoy el sistema), el **cociente de medias E[num]/E[den]** (B, ya guardado como `cociente_medias`) y la **mediana espacial** (C).

| Ratio | Jensen A−B (medio / máx.) | Cambios de categoría A→B | Cambios A→C | Spearman A−B |
|---|---|---|---|---|
| Theta/Alfa | 21.5 % / 133.1 % | 26/156 | 25/156 | 0.973 |
| Delta/Alfa | 38.1 % / 180.8 % | 25/156 | 26/156 | 0.980 |
| Theta/Beta | 28.5 % / 243.8 % | 13/156 | 9/156 | 0.953 |
| Lentificación | 33.5 % / 201.4 % | 30/156 | 36/156 | 0.979 |

**Decisión: NO cambiar producción.** El sesgo es sistemático y grande en valor absoluto, pero (a) el ordenamiento se conserva (Spearman > 0.95), (b) no hay grupo control en la base actual (156/156 son DCL), por lo que no puede demostrarse mejor sensibilidad/especificidad, y (c) los umbrales EO están calibrados contra el estimador A. Cambiar el estimador exigiría recalibrar los cuatro umbrales con una muestra de controles. Queda como línea de investigación; `cociente_medias` se sigue guardando para auditoría.

### Mejora 1 — Protocolo de adquisición (2026-10-19, sin commit)

Objetivo: reducir canales atípicos e interferencia para que la Confiabilidad de los biomarcadores suba de NULA a MEDIA/ALTA. Se procesaron los 154 registros en una carpeta aislada (sin tocar `CACHE` ni informes) para saber qué corregir.

**Qué se añadió al informe PDF** (sección *Calidad técnica*, bloque nuevo **"Recomendaciones de adquisición"**, justo después de "Calidad técnica y canales para revisión"):
- Confiabilidad actual del registro y **cuántos electrodos hay que corregir para llegar a MEDIA y a ALTA** (con sus nombres), usando la MISMA fórmula `_calcular_confiabilidad_metrica`. Si el SNR lo impide, lo dice ("Ni MEDIA ni ALTA se alcanzan solo corrigiendo electrodos…" / "ALTA no se alcanza…").
- **Interferencia de red (60 Hz):** porcentaje del ruido 40-100 Hz que es red, SNR actual vs. SNR sin red y la Confiabilidad que tendría el registro sin esa interferencia (p. ej. DCL 004: 91 % del ruido, SNR −4.3 → 5.6 dB, NULA → MEDIA).
- Tabla **Electrodo / Nivel (Atípico/Sospechoso) / Problema detectado** (hasta 12 filas, ordenadas por severidad).
- **Acciones numeradas para la próxima toma**: red eléctrica (GND común, igualar impedancia REF/GND, alejar equipos), patrón sistémico si aplica, y la acción por cada causa detectada.
- Las recomendaciones aplican a la próxima adquisición; **no modifican el análisis del registro actual**.

**Código:**
- `calcular_snr_por_canal()` guarda además `pct_ruido_red` (mediana entre canales de la fracción 48-52 + 58-62 Hz dentro de 40-100 Hz) y `snr_sin_red_db`. `snr_global_db` no cambia. Desde T-4, `snr_sin_red_db` es la fuente por defecto de la Confiabilidad (ver T-4).
- Nuevos `ACCIONES_ADQUISICION` (causa → título y acción), `UMBRAL_PATRON_SISTEMICO = 0.20`, `UMBRAL_INTERFERENCIA_RED_PCT = 30`, `REDES_ELECTRICAS_HZ = (50, 60)`.
- `diagnosticar_adquisicion()` (electrodos a revisar, causas, patrón sistémico, mínimos para MEDIA/ALTA, interferencia de red) se guarda en `resumen_calidad.json → recomendaciones_adquisicion`. `recomendaciones_adquisicion_desde_cache()` la reconstruye desde `reporte_canales_atipicos.txt` en cachés antiguas (sin datos de red).
- GUI: bloque **"Adquisición · próxima toma"** en el inspector (resumen MEDIA/ALTA, aviso de red/patrón sistémico, hasta 8 electrodos).
- **Fix ICA:** si un canal acapara > 99 % de la varianza, `n_components=0.99` dejaba 1 componente y `ica.fit` abortaba (DCL 099). Ahora reintenta con `min(n_eeg − 1, 30)` componentes.

**Hallazgos en 151 registros válidos:**
- Confiabilidad: ALTA 5 · MEDIA 25 · BAJA 36 · NULA 85. Eliminando solo la red: ALTA 6 · MEDIA 41 · BAJA 36 · NULA 68.
- Red de 60 Hz ≥ 30 % del ruido en 100/151 registros (mediana 64 % en ellos).
- **FZ y F2 fallan en el 95 % de los registros** (2.5 años), son contiguos en el conector y se parecen entre sí (r ≈ 0.85) → sospecha de falla de gorro/cable/pin, no de preparación. Siguiente: CP2 64 %, FCZ 57 %, FC2 51 %.
- Mediana de electrodos a corregir: 7 para MEDIA, 11 para ALTA. En 101 registros el SNR impide ALTA aunque se corrijan todos.
- Patrón sistémico (≥ 20 % de canales con la misma causa) en 46 registros; se mantuvo el umbral para no volver ruidosa la alerta.
- Archivos inservibles: `DCL 026` y `DCL 084` (22 canales en vez de 64), `DCL 042` (`.dap` corrupto, NumSamples = −1).

**Decisión:** las mejoras 2-4 (base normativa propia, modelo de riesgo combinado, seguimiento longitudinal) se **retiraron por completo** del código por decisión del equipo; no se realizarán por ahora. El PSD queda como está.

### T-1 — Pruebas del diagnóstico de adquisición (2026-10-20)

Primera tarea del plan de recomendaciones derivado de la Mejora 1. Se creó `tests/test_adquisicion.py` con **30 pruebas** sobre `unittest` (biblioteca estándar: no agrega dependencias a `requirements.txt`). No procesa señales ni toca `CACHE`: usa detalles sintéticos con la misma forma que produce `detectar_canales_atipicos()` y cachés temporales.

Cobertura:
- `_causa_principal_canal()`: las cinco causas (amplitud, canal bajo, ruido HF, alfa dominante, espectro anómalo), el factor 0.9 del espectro, el fallback sin razón y que toda causa tenga título/acción en `ACCIONES_ADQUISICION`.
- `detalle_atipicos_desde_reporte()`: secciones fuerte/sospechoso, razones encadenadas con ` | `, inferencia de razón en sospechosos, canales fuera de rango, reporte vacío y archivo inexistente.
- `diagnosticar_adquisicion()`: registro sin atípicos, patrón sistémico en el umbral del 20 %, umbral de red del 30 % (incluido el límite exacto), texto SNR con/sin red, mínimos reales para MEDIA/ALTA (caso 32/64 fuertes y SNR 16 dB → 20 y 27 electrodos), `None` cuando el SNR impide el objetivo, FZ+F2 sin patrón, orden fuertes→sospechosos por score y acciones sin duplicados.
- `recomendaciones_adquisicion_desde_cache()`: prioridad del JSON, reconstrucción desde reporte en cachés antiguas, JSON corrupto y caché sin datos.

**Resultado:** 30/30 pruebas OK y `py_compile` OK (`VERIFICACION_SINTACTICA_OK`). Ejecución: `python -m unittest discover -s tests -p "test_*.py" -v`.

**Aprendizaje:** en `reporte_canales_atipicos.txt` el canal va en base 1 (`Canal 07` = índice 6 = `F5`); una expectativa de fixture asumió `F7` y la prueba lo detectó. La alerta de hardware de cohorte FZ/F2 quedó implementada en T-3; la prueba deja documentado que hoy 2/64 no dispara el patrón sistémico del 20 %.

### T-2 — Umbrales de adquisición configurables (2026-10-20)

Los dos umbrales del diagnóstico de adquisición dejaron de estar hardcodeados. Ahora se leen de `config.json` (raíz del proyecto), sin cambiar sus valores por defecto:

```json
{
  "adquisicion": {
    "umbral_interferencia_red_pct": 30.0,
    "umbral_patron_sistemico": 0.2
  }
}
```

- **Cargador:** `cargar_config_adquisicion(ruta=None)` devuelve `(config, origen, advertencias)`. Nunca lanza excepción: si el archivo falta, está corrupto o trae un valor inválido, usa el default y anota el motivo en `CONFIG_ADQUISICION_ADVERTENCIAS`.
- **Validación:** tipo numérico (rechaza `bool`, texto, `null`), finito (rechaza `NaN`/`Infinity`), rango inclusivo `[0, 100]` para el porcentaje y `[0, 1]` para la fracción; un entero JSON desmesurado que desborda `float()` también cae al default. Claves desconocidas se ignoran con advertencia.
- **Formato tolerante:** JSON parcial (mezcla defaults), objeto vacío, BOM de UTF-8 (Bloc de notas) y claves que empiezan por `_` como documentación embebida.
- **Compatibilidad:** las constantes históricas `UMBRAL_INTERFERENCIA_RED_PCT` y `UMBRAL_PATRON_SISTEMICO` se asignan desde `CONFIG_ADQUISICION`, por lo que `diagnosticar_adquisicion()`, el PDF y la GUI siguen leyendo lo mismo sin cambios.
- **Búsqueda del archivo:** en desarrollo, junto a `pipeline_eeg_completo.py`; en el `.exe`, primero junto a `NeuroX.exe` (editable por el técnico) y luego en `_internal` (`sys._MEIPASS`). `NeuroX.spec` empaqueta `config.json` como default.
- **Alcance:** un cambio de umbral aplica al **procesamiento nuevo** y a la reconstrucción de cachés antiguas; los registros ya procesados conservan las recomendaciones guardadas en su `resumen_calidad.json`.

**Pruebas:** `tests/test_config_adquisicion.py` (34 pruebas): integridad del esquema, defaults históricos, JSON válido/parcial/vacío/BOM, claves de documentación, JSON corrupto, raíz/sección no objeto, claves desconocidas, tipos incorrectos, `NaN`/`Infinity`, enteros desmesurados, rangos y límites inclusivos, ruta que es directorio, prioridad de rutas en `.exe` vs `_internal` y cableado real con `diagnosticar_adquisicion()`.

**Resultado:** 64/64 pruebas OK (30 de T-1 + 34 de T-2) y `VERIFICACION_SINTACTICA_OK`.

**Resuelto en T-6:** las advertencias de `config.json` se muestran en el informe PDF, en el inspector y en el log de procesamiento (ver T-6).

### T-3 — Alerta de cohorte FZ/F2 (2026-10-20)

FZ y F2 comparten pines contiguos en el conector. Si fallan como canal atípico fuerte en casi todos los registros, el problema es del gorro/cable/pin y no de la preparación de cada paciente. T-3 convierte esa observación en un aviso automático, sin reprocesar señales:

- **Historial persistente:** `<carpeta_datos>/cohorte_fz_f2.json`. Guarda, por registro (clave = nombre de la carpeta de `CACHE`), qué canales de interés fallaron y en qué nivel:
  ```json
  {"version": 1, "actualizado": "2026-10-20T10:50:24",
   "registros": {"DCL_003": {"FZ": "fuerte", "F2": "fuerte"},
                 "DCL_004": {"FZ": "sospechoso", "F2": "fuerte"}}}
  ```
  Solo conserva `FZ`/`F2`; el nivel `fuerte` prevalece sobre `sospechoso`. Es **idempotente**: reprocesar un archivo reemplaza su entrada y no infla los conteos.
- **Reconstrucción inicial:** si el historial no existe (o está corrupto), se arma leyendo los `calidad_senal/resumen_calidad.json` ya presentes en `CACHE`. No requiere reprocesar. Un historial válido no se vuelve a escanear.
- **Escritura segura:** `tmp` + `os.replace` (atómico en Windows), serializada con `threading.Lock` para que dos hilos no pierdan actualizaciones. Si la escritura falla, el archivo anterior queda intacto y la alerta se evalúa igualmente en memoria.
- **Criterio de alerta:** se exige que **ambos** canales superen el umbral de fallo fuerte (no basta uno) y que haya al menos `min_registros_cohorte` registros. Los sospechosos no cuentan. Texto descriptivo: “patrón de cohorte, no de preparación individual”; nunca afirma diagnóstico ni certeza de hardware.
- **Configuración** (`config.json → adquisicion`): `umbral_patron_cohorte_fz_f2` (fracción 0–1, default `0.80`) y `min_registros_cohorte` (entero, default `20`, rango 1–100000). El loader de T-2 ahora valida enteros: `20.5` se rechaza con advertencia y cae al default.
- **Integración:** `diagnosticar_adquisicion()` devuelve `alerta_cohorte` como campo propio (no se duplica en `acciones_protocolo`). El PDF imprime “Aviso de cohorte FZ/F2” y el inspector usa un texto corto. `recomendaciones_adquisicion_desde_cache()` inyecta la alerta vigente si la caché vive dentro de `<base>/CACHE`; no modifica el `resumen_calidad.json` original.
- **Alcance:** la alerta describe la cohorte de registros procesados, sin ventana temporal; no sustituye la inspección del hardware ni cambia las métricas qEEG del registro.

**Hallazgo real (136 registros con calidad completa):** FZ falla como fuerte en **121/136 (89 %)** y F2 en **124/136 (91 %)**. La alerta se activa con los defaults. El historial quedó creado en `Ejecutar/cohorte_fz_f2.json` (12 KB).

**Pruebas:** `tests/test_cohorte_fz_f2.py` (52 pruebas): rutas/config, carga tolerante (BOM, corrupto, estructura inválida, solo FZ/F2), actualización idempotente y reemplazo, guardado atómico y conservación del original ante fallo, reconstrucción desde cachés, recuperación de historial corrupto, evaluación (mínimo, un solo canal, límite 80 %, sospechosos), registro por hilo, inyección en cachés y extensión entera de config.

**Resultado:** 116/116 pruebas OK (30 T-1 + 34 T-2 + 52 T-3) y `VERIFICACION_SINTACTICA_OK`.

### T-5 — Regresión end-to-end (2026-10-20)

Protege el flujo completo (`procesar_archivo` → `CACHE` → `generar_informe_desde_cache` → PDF) contra cambios no intencionales de resultados. Todo corre en carpetas temporales: **no toca la `CACHE` ni los informes del usuario**.

- **`tests/regresion_e2e.py`** (utilidades, no se descubre como test):
  - `generar_registro_sintetico()`: par `.dat/.dap` determinista (misma semilla = mismos bytes; FLOAT32, orden SAMP, 64 EEG + 5 auxiliares, 500 Hz, 120 s). Fondo 1/f con vecinos correlacionados, alfa de 10 Hz con **gradiente antero-posterior suave**, theta frontal, parpadeos en FP/AF, red de 60 Hz y tres defectos inyectados: **T7** ruido de alta frecuencia, **PO7** casi plano (×0.02), **C4** electrodo suelto (×5 + ruido).
  - `extraer_huella()`: resumen plano de la caché: bandas globales y regionales, ratios (media de cocientes y `cociente_medias`), semáforos, PAF global/occipital, gradiente AP, alfa occipital/frontal, canales atípicos/sospechosos, estado global, Confiabilidad, SNR, `pct_ruido_red`, `snr_sin_red_db`, varianza ICA y recomendaciones de adquisición.
  - `comparar_huellas()`: tolerancias por tipo (bandas ±0.5 pp global y ±1.0 pp regional, ratios y gradiente ±3 % relativo, PAF ±0.15 Hz, SNR ±0.1 dB, red ±0.5 pp, ICA ±2 pp). **Categorías, listas de canales y conteos se comparan exactos.**
- **`tests/test_regresion_e2e.py`** (19 pruebas): comparador (7), generador determinista y legible por `cargar_datos_eeg` (2), flujo completo del sintético (9: estructura de caché, huella = referencia, detección de T7/PO7/C4, PAF ≈ 10 Hz, alfa occipital > frontal, bandas ≈ 100 %, red ≥ 30 %, PDF con sus secciones, nada fuera del temporal) y registros reales (1, opcional).
- **Referencias:** `tests/referencias/e2e_sintetico.json` y `tests/referencias/e2e_reales.json` (huellas de `DCL 003/007/024/083/138`, solo métricas derivadas; no contienen señal).
- **Uso:**
  - Normal (~1 min): `python -m unittest discover -s tests -p "test_*.py" -v`
  - Con registros reales (~5 min): definir `NEUROX_DATOS_REGRESION=<carpeta datosTodos>`.
  - Saltar las pruebas lentas: `NEUROX_SALTAR_E2E=1`.
  - **Regenerar referencias solo tras un cambio intencional de resultados**: `python tests/regresion_e2e.py --actualizar [--reales <carpeta>]`, y explicar en el commit por qué cambian.

**Resultado:** 135/135 pruebas OK (116 previas + 19 de T-5), incluidos los 5 registros reales. **Verificación de sensibilidad:** mover el inicio de Alfa de 8.0 a 8.5 Hz (en memoria) produjo 15 diferencias detectadas (Alfa global −1.5 pp), así que la regresión no es complaciente.

**Aprendizajes del sintético:**
- El detector **nunca marca como fuerte un canal solo por amplitud** (diseño intencional: exige además rareza espectral). Un canal ×6 con espectro normal no se detecta.
- La parte del alfa común a todos los canales la elimina el CAR; para subir el alfa hay que darle estructura espacial.
- Un alfa posterior concentrado en pocos canales hace que el detector los marque como `alfa_dominante` (lo mismo ocurre en registros reales: es la 4.ª causa de fallo). Un gradiente suave lo evita.
- Elegir valores lejos de los cortes clínicos: con Lentificación = 1.342 (corte 1.35) cualquier variación numérica entre equipos cambiaría el semáforo y la prueba sería inestable.

### T-4 — SNR sin red eléctrica en la Confiabilidad (2026-10-23)

**Decisión del equipo:** "Calidad para diagnóstico = SÍ". La Confiabilidad de biomarcadores debe medir la calidad del análisis, no solo la calidad de la toma.

- **Problema:** la Confiabilidad usaba `snr_global_db` (1-30 vs 40-100 Hz), que incluye la red de 50/60 Hz. Los biomarcadores, en cambio, se calculan sobre la señal ya filtrada con notch. Un registro con buena señal y una toma de tierra deficiente quedaba castigado como si el análisis fuera malo.
- **Solución:** `calcular_snr_por_canal()` ya guardaba `snr_sin_red_db` (mismo SNR pero descontando la potencia de 50/60 Hz del ruido). T-4 lo convierte en la fuente por defecto de `confiabilidad_biomarcadores` mediante:
  - `config.json → adquisicion.usar_snr_sin_red_en_confiabilidad` (booleano, default `true`);
  - `USAR_SNR_SIN_RED_EN_CONFIABILIDAD` y `_snr_para_confiabilidad()`, que devuelve `(snr_db, fuente)` y cae a `snr_global_db` si el valor falta, no es finito o el flag está en `false`;
  - bloque `confiabilidad_snr` en `resumen_calidad.json`: `{"fuente", "snr_db", "incluye_ruido_red"}`, para auditar qué SNR se usó en cada registro.
- **Coherencia con las recomendaciones:** `diagnosticar_adquisicion()` usa la misma fuente que la Confiabilidad persistida (`confiabilidad_snr_fuente`). El parámetro `usar_snr_sin_red=None|True|False` permite forzar la fuente al reconstruir cachés antiguas. Con el flag activo, el aviso de red ya no promete una mejora que la métrica tiene aplicada: informa que la Confiabilidad ya excluye ese ruido y que corregir GND/REF sigue siendo recomendable.
- **Compatibilidad:** las cachés ya procesadas conservan el valor guardado (no se reprocesan). Si una caché no tiene `confiabilidad_biomarcadores`, el respaldo del PDF/Excel sigue usando el SNR crudo histórico, para que ambos entregables coincidan. Los registros nuevos muestran "(SNR sin red eléctrica)" junto a la Confiabilidad del PDF.
- **Efecto medido en los 5 registros de regresión:** DCL 024 BAJA→MEDIA, DCL 138 NULA→BAJA, DCL 007 requiere 4 electrodos en vez de 8 para MEDIA; DCL 003 y DCL 083 no cambian. Ningún registro empeoró. En el sintético, ALTA pasa a ser alcanzable corrigiendo 1 electrodo (antes inalcanzable).
- **Pruebas:** `tests/test_snr_confiabilidad.py` (17): selección de fuente, valores no finitos, validación booleana estricta del config, coherencia del diagnóstico, texto de red con/sin flag, snapshot de caché y tolerancia a JSON corrupto.

### T-6 — Configuración de adquisición trazable (2026-10-23)

Los umbrales de T-2/T-3 podían cambiar sin que el informe dijera con cuáles se calculó. T-6 hace visible y auditable la configuración aplicada, sin reprocesar señales.

- **Snapshot en la caché:** `resumen_calidad.json → config_adquisicion` guarda `ruta`, `umbrales` (los 5, incluido el flag de T-4) y `advertencias_carga` del cargador. Es la configuración que realmente se usó al procesar ese registro.
- **`config_adquisicion_desde_cache(carpeta)`:** devuelve el snapshot si existe; si la caché es anterior, devuelve la configuración vigente con `guardada_en_cache=False`. Revalida los valores leídos del disco, así que una caché editada a mano no rompe el informe. Nunca lanza excepción.
- **Informe PDF**, bloque "Configuración de adquisición aplicada" dentro de *Calidad técnica*, tras las recomendaciones: origen (`config.json` o defaults), recuadro ámbar con las advertencias del cargador, tabla Parámetro/Valor (interferencia de red, patrón sistémico, cohorte FZ/F2, mínimo de registros y SNR de la Confiabilidad) y nota cuando la caché no guardó el snapshot. En cachés antiguas la fila de SNR dice "No registrado (caché anterior: usó SNR crudo)" en vez de atribuirle la política nueva.
- **Inspector (GUI):** línea `Config: red 30 % · sistémico 20 % · cohorte 80 % · mín. 20 · SNR sin red`, con la marca "(vigente; caché sin snapshot)" cuando corresponde. Las advertencias de `config.json` se añaden al recuadro amarillo del bloque, y se muestran aunque la caché aún no tenga recomendaciones.
- **Log de procesamiento:** al iniciar `procesar_archivo()` se registra la ruta de `config.json` (o el uso de defaults) y cada advertencia del cargador, visibles en el panel de log de la GUI.
- **Pruebas:** incluidas en `tests/test_snr_confiabilidad.py` (7 de T-6): caché sin snapshot, snapshot guardado, revalidación de valores corruptos, JSON corrupto y carpeta inexistente; más las aserciones del PDF y del snapshot en `test_regresion_e2e.py`.

### Higiene del repositorio (2026-10-23)

- **Pruebas fuera de Git:** `tests/` se retiró del índice (`git rm --cached`) y se añadió a `.gitignore`. Los archivos siguen en disco en `proyecto-NeuroX/tests/` (y se copiaron a la raíz para poder ejecutar la suite completa desde el directorio de trabajo). El historial conserva las versiones de T-1, T-2, T-3 y T-5; se pueden recuperar con `git checkout <commit> -- tests`.
- **Datos fuera de Git:** `Ejecutar/` también salió del índice (incluía `DCL 003.dap` y un PDF de ejemplo, ~5 MB de registro médico). Se ignoran además `*.dap`, `*.rs3`, `*.ceo`, `*.bdf`, `*.edf` y `*.log`, que antes podían colarse porque solo `*.dat` estaba excluido.
- **Temporales eliminados de la raíz:** `comprehensive_fix.py`, `pipeline_eeg_completo.py.git`, `pipeline_eeg_completo.py.original`, `proyecto-NeuroX_BACKUP_20261014_0925/` y los `__pycache__/`.
- **Verificación:** suite completa desde la raíz del proyecto → **135 pruebas OK, 1 omitida** (la de registros reales, que exige `NEUROX_DATOS_REGRESION`), en ~20 s; código de salida 0.
- **Lección:** `git rm --cached` deja los archivos locales intactos; fue la vía segura para dejar de versionar datos clínicos sin borrarlos.

### Identidad de commits

Los 14 commits históricos de la identidad `KEVIN ALEXANDER YEPEZ OLIVEROS <kevin.yepez@unisimon.edu.co>` se reescribieron a `Kypez-dev <322148154+Kypez-dev@users.noreply.github.com>` (correo *noreply* oficial de la cuenta GitHub `Kypez-dev`, ID 322148154), con `git filter-branch` y `--force-with-lease`. Respaldo del historial anterior en la rama `backup/pre-rewrite-20261019`. La configuración `user.name`/`user.email` quedó fijada en local y global (`user.useConfigOnly = true`) para que también la use VS Code.

---

## Aprendizajes y Errores a Evitar
- **Saturación por sobre-threading en librerías BLAS:**
  - *Problema:* El uso simultáneo de `ThreadPoolExecutor` en Python junto con las operaciones matriciales internas multihilo de OpenBLAS/MKL saturaba la CPU al 100% congelando la UI.
  - *Solución implementada:* Forzar `OMP_NUM_THREADS="1"`, `MKL_NUM_THREADS="1"`, etc., antes de importar NumPy. Mantener siempre este bloque al principio de `pipeline_eeg_completo.py`.
- **Bloqueo de ficheros binarios en Windows (`np.memmap`):**
  - *Problema:* Al reprocesar un archivo que ya estaba abierto en el visor, Windows lanzaba `PermissionError` porque el archivo `.dat`/`.npy` seguía retenido por la memoria virtual.
  - *Solución implementada:* Crear la rutina `_liberar_cache_visual()` en la GUI, eliminando referencias explícitas y ejecutando `gc.collect()` antes de iniciar un reprocesamiento.
- **Sobre-interpolación destructiva de la varianza:**
  - *Problema:* Al interpolar tanto canales "fuertes" como "sospechosos", en algunos montajes ruidosos se reemplazaban hasta 20 de los 64 canales, aplanando la actividad cortical real.
  - *Solución implementada:* La interpolación IDW se reservó únicamente para canales clasificados como "fuertes". Los canales sospechosos solo se excluyen del cálculo del promedio CAR.
- **Ruptura de compatibilidad con NumPy 2.0:**
  - *Problema:* La función `np.trapz` fue retirada en NumPy 2.0, provocando excepciones fatales en el cálculo del área bajo la curva PSD.
  - *Solución implementada:* Creación del wrapper polifill `_trapz` con fallback dinámico a `np.trapezoid` o `np.trapz`.
- **Fallas de renderizado en PDF por formatos de imagen:**
  - *Problema:* `fpdf2` lanza excepciones críticas si una imagen no es un PNG estrictamente estándar (ej. paletas con transparencias no convencionales).
  - *Solución implementada:* Función `_normalizar_imagen_para_pdf` que valida y recomprime las figuras mediante PIL antes de entregárselas al maquetador PDF.
- **Bug del límite inferior en `clasificar_semaforo` (falsos "Patologico"):**
  - *Problema:* Al evaluar ratios con `low_normal <= valor <= high_normal`, un sujeto muy sano con un cociente bajo (p. ej. Theta/Alfa = 0.18) no entraba en `normal` ni en `borderline` y caía en el `else` → clasificado en rojo.
  - *Solución implementada:* Se llevó el piso de todos los ratios de lentificación a `0.00` en `REFERENCIA_RATIOS_EO`, de modo que valores bajos sean "Normal" y no "Patologico".
- **Desigualdad de Jensen en el promediado de cocientes:**
  - *Problema:* `calcular_ratios_welch_principales` divide por canal y luego promedia los cocientes; matemáticamente E[D/A] > E[D]/E[A], inflando los ratios globales (p. ej. Delta 21.7% / Alfa 28.1% daba DAR ≈ 1.12 en vez de 0.77).
  - *Solución implementada:* Se documenta el sesgo y los cortes EO lo absorben; se guarda `cociente_medias` como métrica de auditoría, sin alterar el cálculo principal.
- **Contaminación de bandas por el acondicionamiento del pipeline (Delta inflado, Alfa/Beta encogidos):**
  - *Problema (6 causas concurrentes):* (1) el SNR adaptativo de ICA se medía sobre la señal ya recortada a 40 Hz, con la banda de ruido 40-100 Hz vacía, de modo que el SNR salía artificialmente alto y el modo agresivo nunca se activaba; (2) ICLabel recibía solo 1-40 Hz aunque fue entrenado con 1-100 Hz, perdiendo la información de alta frecuencia para separar músculo/parpadeo; (3) el soft-thresholding Wavelet se aplicaba a TODOS los detalles, encogiendo Alfa/Beta y elevando Delta relativo; (4) las épocas contaminadas se recortaban y re-pegaban (`datos_fn[:, mask]`), creando saltos verticales y energía falsa de baja frecuencia; (5) Welch usaba media aritmética (contaminada por unas pocas ventanas transitorias de alta potencia en Delta); (6) los electrodos M1/M2/CB1/CB2 (marcados 'misc', no limpiados por ICA) entraban en los promedios corticales globales.
  - *Solución implementada:* copia de banda ancha 1-100 Hz usada SOLO para el fit de ICA e ICLabel; máscara robusta (Mediana/MAD) usada SOLO para `ica.fit`, con `ica.apply` sobre la señal continua completa; Wavelet limitado a los detalles finos (`WAVELET_DETALLES_LIMPIAR`); Welch `average='median'` con ventana de 2 s; exclusión de periféricos y atípicos en los promedios globales (`calcular_potencia_relativa_welch`).
  - *Evidencia:* en 14 registros reales el Delta global bajó en TODOS (media 51.5% → 44.8%), con Alfa subiendo en casi todos y sin ningún empeoramiento (p. ej. DCL_138 39.5→20.1; DCL_007 55.1→46.4).
  - *Lección:* una ventana Welch de 4 s desplazaba el Delta al alza de forma no fisiológica; la combinación correcta es 2 s + mediana. La mejora es de robustez frente a artefactos, no una "normalización" forzada: los registros genuinamente ruidosos siguen con Delta alto (correcto).
- **Falso "Lentificado" por tomar el máximo absoluto (argmax) sobre la rampa 1/f (PAF):**
  - *Problema:* `calcular_paf_por_canal` usaba el máximo absoluto dentro de 8-13 Hz. Con ojos abiertos y sin pico Alfa, el espectro solo desciende (ley 1/f) y el argmax cae en el borde inferior de la banda (8.0-8.5 Hz), reportando una lentificación inexistente. El patrón se repitió (muchos registros con PAF clavado en 8.5 Hz) y en el Excel un PAF `NaN` (sin pico) caía en el `else` y se rotulaba "Lentificado" (rojo).
  - *Solución implementada:* El PAF descuenta la pendiente 1/f local (recta que une los extremos 8-13 Hz) y exige crestas reales con `scipy.signal.find_peaks` + umbral de prominencia (`PAF_EO_PARAMS`). Si no hay cresta válida → canal inválido. Un PAF no finito se rotula `ESTADO_PAF_SIN_PICO = "Bloqueo visual (Normal EO)"` (verde) en PDF, GUI y Excel, nunca "Lentificado". Calibrado sobre 156 registros: conserva picos genuinos (DCL_003 en 8.5 Hz = lentificación real) y descarta rampas/hombros (DCL_024 occipital → sin pico).

---

## Decisiones de Arquitectura (y por qué)
- **Corte inferior del filtro pasa-banda en 1.0 Hz (estándar IFCN/ACNS):**
  - *Por qué:* Elimina artefactos no-fisiológicos < 1 Hz (derivas por sudor, polarización de electrodos, potenciales de piel, artefactos respiratorios) que inflaban la banda Delta 3-10x por encima de lo normal. Alinea el filtro con el análisis Welch (1-30 Hz) y con ICLabel (1-100 Hz). La banda Delta (1-4 Hz) se caracteriza completamente sin necesidad de incluir frecuencias sub-Delta.
- **Escala de Densidad Espectral en Decibelios (dB) y normalización visual:**
  - *Por qué:* El cerebro humano sigue fisiológicamente la "Ley de Potencia $1/f$". En escala lineal pura, las bandas Delta dominan toda la gráfica y dejan a Alfa y Beta invisibles. La escala logarítmica y el escalado por percentiles en el visor permite interpretar todas las bandas cognitivas.
- **Aislamiento de MNE y ICLabel en copia de señal temporal:**
  - *Por qué:* ICLabel fue entrenado con señales referenciadas a CAR y filtradas a 1–100 Hz. Para evitar modificar la física del registro crudo de entrada, esa configuración solo se aplica a una instancia temporal (`raw_fit`) que entrena el ICA; luego, la matriz de proyección resultante se aplica de forma limpia a la señal de trabajo.
- **Uso de ventanas no modales para consolidación de reportes:**
  - *Por qué:* Evita el bloqueo modal (`grab_set`), permitiendo que el especialista continúe explorando señales individuales en la interfaz mientras se procesan consolidaciones masivas en segundo plano.
- **Referencias clínicas separadas por condición de registro (EO vs EC):**
  - *Por qué:* Los registros de NeuroX son de reposo con ojos abiertos y fijación (20-60 años). La reactividad de Berger reduce la amplitud de Alfa 40-70%, lo que infla el porcentaje relativo de Delta y los cocientes; aplicar umbrales de ojos cerrados generaba falsos positivos. La frecuencia del PAF (no su amplitud) es estable, por eso PAF mantiene sus umbrales. Se conservan los rangos EC para compatibilidad y auditoría retrospectiva.
- **Umbrales en un único lugar por entregable:**
  - *Por qué:* Pipeline y GUI comparten `clasificar_semaforo()`; el Excel no importa el pipeline (para evitar arrastrar dependencias pesadas) y replica solo PAF/Gradiente AP, con un comentario que obliga a mantenerlos sincronizados si cambian.
- **Clasificar canales periféricos ANTES de los prefijos:**
  - *Problema:* `startswith("C")`/`startswith("CB")` capturaba `CB1`/`CB2` y los metía en Central/Occipital. Como ICA los trata como `misc` (no los limpia), contaminaban promedios regionales y topomaps.
  - *Regla:* resolver primero la lista explícita de periféricos (`M1`, `M2`, `CB1`, `CB2`) y recién después evaluar prefijos. Cualquier función nueva de regionalización debe replicar este orden.
- **Verificar la referencia espacial, no solo el filtro, antes de entrenar ICA:**
  - *Problema:* entrenar con la media de los 60 canales EEG y aplicar la matriz sobre una señal con CAR robusto (que excluye sospechosos) desalinea el subespacio espacial: la proyección deja de ser la inversa del mezclado.
  - *Regla:* `datos_ica_fit` debe construirse con el MISMO `aplicar_car_robusto()` que `datos_fn`. No volver a llamar `set_eeg_reference('average')` sobre `raw_fit`.
- **Métricas de adquisición se miden antes de referenciar:**
  - *Problema:* aplicar CAR antes de `_calcular_snr_crudo_rapido()` cambió el SNR de -2.2 dB a 18.8 dB en un registro, alterando el umbral adaptativo de ICLabel (de agresivo a estándar) sin que mediara una mejora real de adquisición.
  - *Regla:* el SNR que alimenta decisiones debe medirse sobre la señal sin referenciar y pasarse explícitamente (`snr_crudo_db=`).

---

## Próximos Pasos

**Pendiente para que la Confiabilidad llegue a MEDIA/ALTA (Mejora 1).** El software ya diagnostica; lo que falta se corrige en la adquisición:
- [ ] **Revisar FZ y F2 en el hardware:** medir continuidad del electrodo, cable y pin del conector, o probar con otro gorro. T-3 ya lo detecta automáticamente (121/136 y 124/136 con los defaults); mientras no se reparen seguirán saliendo atípicos en casi todas las tomas.
- [ ] **Eliminar la interferencia de red:** GND del paciente y del amplificador al mismo punto, impedancia de REF/GND igual a la de los demás, alejar cargadores/monitores/lámparas, no cruzar cables de electrodos con los de alimentación. Es la corrección de mayor impacto (17 registros saldrían de NULA sin tocar electrodos).
- [ ] **Aplicar las acciones del informe antes de cada toma** (reabrasionar y renovar gel en electrodos con ruido HF, limpiar puentes de gel, fijar cables) y verificar impedancias antes de grabar.
- [ ] **Medir el efecto:** tras corregir hardware y red, grabar nuevos registros y comparar la distribución de Confiabilidad contra la línea base (5/25/36/85).
- [x] **SNR de la Confiabilidad (decidido):** el equipo eligió "Calidad para diagnóstico = SÍ". Desde T-4 la Confiabilidad usa `snr_sin_red_db` (sin la red de 50/60 Hz) por defecto, coherente con los biomarcadores, que se calculan sobre la señal con notch. El SNR crudo sigue visible en el informe y en las recomendaciones de adquisición.
- [ ] Volver a adquirir o descartar `DCL 026`, `DCL 084` y `DCL 042`.
- [x] Commit de la Mejora 1 (`9452708`).

**Plan de recomendaciones (T-1…T-6):**
- [x] **T-1 — Tests del diagnóstico de adquisición:** `tests/test_adquisicion.py`, 30/30 OK.
- [x] **T-2 — Parametrizar umbrales** (`config.json`): loader tolerante + 34 pruebas; defaults 30.0 / 0.20 intactos.
- [x] **T-3 — Alerta de cohorte FZ/F2** (posible hardware del conector) en pipeline, PDF e inspector: historial persistente e idempotente, reconstrucción desde `CACHE`, 52 pruebas. Detecta 89 %/91 % en los 136 registros actuales.
- [x] **T-4 — `snr_sin_red_db` en la Confiabilidad:** flag `usar_snr_sin_red_en_confiabilidad` (default `true`), inyectado en `resumen_calidad.json` y en las recomendaciones; 17 pruebas nuevas. Ver T-4.
- [x] **T-5 — Regresión end-to-end:** registro sintético + 5 reales contra huellas de referencia (`tests/test_regresion_e2e.py`, 20 pruebas; 154/154 OK con reales). Referencias regeneradas por T-4.
- [x] **T-6 — Documentar configuración y umbrales:** sección "Configuración de adquisición aplicada" en el PDF, línea de config + advertencias en el inspector y snapshot en la caché. Ver T-6.

**Otros:**
- [ ] **Reprocesamiento masivo pendiente:** las ~153-156 entradas de `CACHE` y sus informes aún reflejan el pipeline anterior. Con S-01 el costo bajó a ~15-25 s/archivo; estimación total 1-1.5 h más PDFs y consolidado. Respaldar/renombrar la `CACHE` actual antes de empezar.
- [ ] **Fase 5 (hipótesis H-01):** experimento aislado de normalización por percentiles; no adopción en producción sin evidencia empírica.
- [ ] Validar el build de `NeuroX.spec` con PyInstaller (D-05) y comprobar que ICLabel clasifica dentro del `.exe` (modelo ONNX empaquetado).
- [ ] Explorar el cálculo de asimetría interhemisférica frontal (biomarcador complementario en investigación de deterioro cognitivo).