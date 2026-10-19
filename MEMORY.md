# MEMORY.md — NeuroX

## Estado Actual
- **Sincronización (2026-10-19):** la raíz del proyecto y `proyecto-NeuroX/` son idénticas, en el commit `b74cc11` (`main == origin/main`). Los commits se firman como `Kypez-dev <322148154+Kypez-dev@users.noreply.github.com>`; el historial previo quedó respaldado en `backup/pre-rewrite-20261019`.
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
- **D-02 — Confiabilidad distinta entre PDF y Excel y con insumos distintos.** Se unificó en una sola fuente de verdad: `calcular_resumen_calidad_senal()` calcula `confiabilidad_biomarcadores` (SNR crudo 1-30 vs 40-100 Hz + % atípicos + % varianza ICA) y la guarda en `resumen_calidad.json`. El PDF y `consolidado_excel.py` la LEEN; solo si falta (caché antigua) recalculan con la misma lógica. La varianza ICA dejó de premiarse (+2 por ≥15 %): un sujeto limpio sin artefactos tiene legítimamente varianza removida baja. Solo penaliza ICA que no removió nada con SNR bajo. Paridad verificada 5/5.
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
- `calcular_snr_por_canal()` guarda además `pct_ruido_red` (mediana entre canales de la fracción 48-52 + 58-62 Hz dentro de 40-100 Hz) y `snr_sin_red_db`. `snr_global_db` no cambia.
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
- [ ] **Revisar FZ y F2 en el hardware:** medir continuidad del electrodo, cable y pin del conector, o probar con otro gorro. Mientras no se reparen seguirán saliendo atípicos en casi todas las tomas.
- [ ] **Eliminar la interferencia de red:** GND del paciente y del amplificador al mismo punto, impedancia de REF/GND igual a la de los demás, alejar cargadores/monitores/lámparas, no cruzar cables de electrodos con los de alimentación. Es la corrección de mayor impacto (17 registros saldrían de NULA sin tocar electrodos).
- [ ] **Aplicar las acciones del informe antes de cada toma** (reabrasionar y renovar gel en electrodos con ruido HF, limpiar puentes de gel, fijar cables) y verificar impedancias antes de grabar.
- [ ] **Medir el efecto:** tras corregir hardware y red, grabar nuevos registros y comparar la distribución de Confiabilidad contra la línea base (5/25/36/85).
- [ ] **Decidir sobre el SNR de la Confiabilidad:** hoy incluye la red de 60 Hz, que el notch elimina antes de calcular los biomarcadores. Usar `snr_sin_red_db` sería más fiel a la calidad real de 1-30 Hz, pero cambia la métrica y exige acuerdo del equipo; por ahora NO se cambió.
- [ ] Volver a adquirir o descartar `DCL 026`, `DCL 084` y `DCL 042`.
- [ ] Commit de los cambios de la Mejora 1 (`Interfaz_neuroX.py`, `pipeline_eeg_completo.py`).

**Otros:**
- [ ] **Reprocesamiento masivo pendiente:** las ~153-156 entradas de `CACHE` y sus informes aún reflejan el pipeline anterior. Con S-01 el costo bajó a ~15-25 s/archivo; estimación total 1-1.5 h más PDFs y consolidado. Respaldar/renombrar la `CACHE` actual antes de empezar.
- [ ] **Fase 5 (hipótesis H-01):** experimento aislado de normalización por percentiles; no adopción en producción sin evidencia empírica.
- [ ] Incorporar tests unitarios automatizados que evalúen la integridad del pipeline matemático usando señales sintéticas predefinidas.
- [ ] Validar el build de `NeuroX.spec` con PyInstaller (D-05) y comprobar que ICLabel clasifica dentro del `.exe` (modelo ONNX empaquetado).
- [ ] Explorar el cálculo de asimetría interhemisférica frontal (biomarcador complementario en investigación de deterioro cognitivo).