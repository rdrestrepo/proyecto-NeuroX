

import os
import json

from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

AZUL_NEUROX = "061A5B"
NARANJA_NEUROX = "FF6A00"
BLANCO = "FFFFFF"

REGIONES = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
BANDAS = ["Delta", "Theta", "Alfa", "Beta"]
ESTADOS_DETALLE = ["Aceptable", "Comprometida", "Deficiente"]  # igual que la plantilla (sin "Buena")

ENCABEZADOS = [
    "Archivo procesado", "Estado calidad", "Duración (s)", "Fs (Hz)",
    "Delta (%)", "Theta (%)", "Alfa (%)", "Beta (%)", "Banda dominante",
    "Theta/Alfa", "Delta/Alfa", "Theta/Beta", "Lentificación",
    "Atípicos (%)", "Sospechosos (%)", "Índice técnico (%)", "SNR (dB)",
    "Observaciones",
]


# ---------------------------------------------------------------------------
# 1. Lectura de los datos ya generados por el pipeline para cada archivo
# ---------------------------------------------------------------------------

def _leer_json_si_existe(ruta):
    if ruta and os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _leer_meta_txt(carpeta_cache):
    meta = {}
    ruta = os.path.join(carpeta_cache, "meta.txt")
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                if "=" in linea:
                    clave, valor = linea.split("=", 1)
                    meta[clave.strip()] = valor.strip()
    return meta


def recolectar_datos_archivo(nombre_archivo, carpeta_cache):
    """
    Lee del CACHE de un archivo ya procesado todo lo necesario para una fila
    del consolidado. Devuelve None si falta información esencial (por
    ejemplo si el archivo no se ha procesado todavía).
    """
    resumen_calidad = _leer_json_si_existe(
        os.path.join(carpeta_cache, "calidad_senal", "resumen_calidad.json")
    )
    resumen_welch = _leer_json_si_existe(
        os.path.join(carpeta_cache, "analisis_welch", "pot_rel_promedios.json")
    )
    resumen_ratios = _leer_json_si_existe(
        os.path.join(carpeta_cache, "analisis_welch", "ratios_promedios_principales.json")
    )

    if not resumen_calidad or not resumen_welch or not resumen_ratios:
        return None

    meta = _leer_meta_txt(carpeta_cache)

    prom_bandas = resumen_welch.get("promedios_globales", {}) or {}
    bandas_valores = {b: prom_bandas.get(b) for b in BANDAS}
    banda_dominante = None
    valores_validos = {b: v for b, v in bandas_valores.items() if isinstance(v, (int, float))}
    if valores_validos:
        banda_dominante = max(valores_validos, key=valores_validos.get)

    ratios = resumen_ratios.get("promedios_globales", {}) or {}
    snr_info = resumen_calidad.get("snr") or {}

    duracion_s = meta.get("duracion_s")
    fs_val = meta.get("fs")

    nombre_registro = os.path.splitext(os.path.basename(str(nombre_archivo)))[0]

    fila = {
        "archivo": nombre_registro,
        "estado": resumen_calidad.get("estado_global"),
        "duracion_s": float(duracion_s) if duracion_s not in (None, "") else None,
        "fs": int(float(fs_val)) if fs_val not in (None, "") else None,
        "delta": bandas_valores.get("Delta"),
        "theta": bandas_valores.get("Theta"),
        "alfa": bandas_valores.get("Alfa"),
        "beta": bandas_valores.get("Beta"),
        "banda_dominante": banda_dominante,
        "theta_alfa": ratios.get("Theta_Alfa"),
        "delta_alfa": ratios.get("Delta_Alfa"),
        "theta_beta": ratios.get("Theta_Beta"),
        "lentificacion": ratios.get("Lentificacion"),
        "atipicos": resumen_calidad.get("porcentaje_atipicos"),
        "sospechosos": resumen_calidad.get("porcentaje_sospechosos"),
        "indice_tecnico": resumen_calidad.get("porcentaje_ponderado"),
        "snr": snr_info.get("snr_global_db"),
        "observaciones": "",
        "potencia_regional": resumen_welch.get("promedios_regionales", {}) or {},
    }
    return fila


def _fila_a_lista(fila):
    return [
        fila["archivo"], fila["estado"], fila["duracion_s"], fila["fs"],
        fila["delta"], fila["theta"], fila["alfa"], fila["beta"],
        fila["banda_dominante"], fila["theta_alfa"], fila["delta_alfa"],
        fila["theta_beta"], fila["lentificacion"], fila["atipicos"],
        fila["sospechosos"], fila["indice_tecnico"], fila["snr"],
        fila["observaciones"],
    ]


# ---------------------------------------------------------------------------
# 2. Utilidades de estilo
# ---------------------------------------------------------------------------

def _estilo_encabezado(ws, celda):
    ws[celda].font = Font(name="Calibri", bold=True, color=BLANCO)
    ws[celda].fill = PatternFill("solid", fgColor=AZUL_NEUROX)
    ws[celda].alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _autoancho(ws, ancho_por_columna):
    for col, ancho in ancho_por_columna.items():
        ws.column_dimensions[col].width = ancho


def _tabla_estilo():
    return TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False,
    )


# ---------------------------------------------------------------------------
# 3. Construcción del libro
# ---------------------------------------------------------------------------

def generar_consolidado_excel(archivos_seleccionados, ruta_salida, logger=None):
    """
    archivos_seleccionados: lista de tuplas (nombre_archivo, carpeta_cache)
        de los .dat que el usuario marcó en la ventana de selección.
    ruta_salida: ruta completa del .xlsx a crear.
    logger: función opcional para mensajes de progreso (mismo patrón que el resto del pipeline).

    Devuelve (ruta_salida, archivos_incluidos, archivos_omitidos).
    """
    def log(msg):
        if logger:
            try:
                logger(msg)
            except Exception:
                pass

    filas = []
    omitidos = []
    for nombre_archivo, carpeta_cache in archivos_seleccionados:
        fila = recolectar_datos_archivo(nombre_archivo, carpeta_cache)
        if fila is None:
            omitidos.append(nombre_archivo)
            log(f"[Aviso] Sin datos de calidad/potencia para '{nombre_archivo}'; se omite del consolidado.")
            continue
        filas.append(fila)

    if not filas:
        raise ValueError(
            "Ninguno de los archivos seleccionados tiene datos procesados "
            "completos (resumen_calidad.json / pot_rel_promedios.json / "
            "ratios_promedios_principales.json). No se generó el consolidado."
        )

    wb = Workbook()
    wb.remove(wb.active)

    ws_resumen = _hoja_resumen(wb, filas)
    _hoja_potencia_regional(wb, filas)
    _hojas_detalle(wb, filas)
    _hoja_potencia_regional_bandas(wb)
    _hoja_estado_calidad(wb, len(filas))
    _hoja_biomarcadores(wb)
    _hoja_calidad_senal(wb)
    _hoja_banda_dominante(wb)

    # Mismo orden de pestañas que la plantilla
    orden = [
        
        "Resumen global por archivo", "Potencia regional", "Estado de calidad",
        "Potencia regional bandas", "Biomarcadores", "Calidad Señal", "Banda dominante",
    ]
    wb._sheets = [wb[nombre] for nombre in orden]
    wb.active = wb.sheetnames.index("Resumen global por archivo")

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)) or ".", exist_ok=True)
    wb.save(ruta_salida)
    log(f"[OK] Consolidado guardado en: {ruta_salida}")
    return ruta_salida, [f["archivo"] for f in filas], omitidos


def _hoja_resumen(wb, filas):
    ws = wb.create_sheet("Resumen global por archivo")

    ws["E1"] = "POTENCIA RELATIVA"
    ws["J1"] = "PROPORCION ENTRE BANDAS (BIOMARCADORES)"
    ws["N1"] = "CALIDAD DE LA SEÑAL"
    for c in ("E1", "J1", "N1"):
        ws[c].font = Font(bold=True, color=BLANCO)
        ws[c].fill = PatternFill("solid", fgColor=NARANJA_NEUROX)

    for idx, encabezado in enumerate(ENCABEZADOS, start=1):
        celda = f"{get_column_letter(idx)}2"
        ws[celda] = encabezado
        _estilo_encabezado(ws, celda)

    fila_inicial = 3
    for i, fila in enumerate(filas):
        r = fila_inicial + i
        for j, valor in enumerate(_fila_a_lista(fila), start=1):
            ws.cell(row=r, column=j, value=valor)

    fila_final = fila_inicial + len(filas) - 1
    ref = f"A2:R{fila_final}"
    tabla = Table(displayName="Tabla1", ref=ref)
    tabla.tableStyleInfo = _tabla_estilo()
    ws.add_table(tabla)

    ws.freeze_panes = "A3"
    _autoancho(ws, {
        "A": 18, "B": 14, "C": 12, "D": 10, "E": 10, "F": 10, "G": 10, "H": 10,
        "I": 14, "J": 10, "K": 10, "L": 10, "M": 12, "N": 10, "O": 12, "P": 14,
        "Q": 10, "R": 20,
    })
    return ws


def _hoja_potencia_regional(wb, filas):
    ws = wb.create_sheet("Potencia regional")
    encabezados = ["Archivo procesado", "Región", "Delta (%)", "Theta (%)", "Alfa (%)", "Beta (%)"]
    for idx, encabezado in enumerate(encabezados, start=1):
        celda = f"{get_column_letter(idx)}1"
        ws[celda] = encabezado
        _estilo_encabezado(ws, celda)

    r = 2
    for fila in filas:
        pot_regional = fila.get("potencia_regional") or {}
        for region in REGIONES:
            valores = pot_regional.get(region)
            if not valores:
                continue
            ws.cell(row=r, column=1, value=fila["archivo"])
            ws.cell(row=r, column=2, value=region)
            ws.cell(row=r, column=3, value=valores.get("Delta"))
            ws.cell(row=r, column=4, value=valores.get("Theta"))
            ws.cell(row=r, column=5, value=valores.get("Alfa"))
            ws.cell(row=r, column=6, value=valores.get("Beta"))
            r += 1

    fila_final = max(r - 1, 2)
    tabla = Table(displayName="Tabla2", ref=f"A1:F{fila_final}")
    tabla.tableStyleInfo = _tabla_estilo()
    ws.add_table(tabla)
    ws.freeze_panes = "A2"
    _autoancho(ws, {"A": 18, "B": 12, "C": 10, "D": 10, "E": 10, "F": 10})
    return ws


def _hojas_detalle(wb, filas):
    """Igual que en tu plantilla: una copia filtrada por estado, en hojas ocultas."""
    tablas = ["Tabla5", "Tabla6", "Tabla7"]
    for i, estado in enumerate(ESTADOS_DETALLE):
        nombre_hoja = f"Detalle{i + 1}"
        ws = wb.create_sheet(nombre_hoja)
        ws["A1"] = f"Detalles para Cuenta de Estado calidad - Estado calidad: {estado}"
        ws["A1"].font = Font(bold=True)

        for idx, encabezado in enumerate(ENCABEZADOS, start=1):
            celda = f"{get_column_letter(idx)}3"
            ws[celda] = encabezado
            _estilo_encabezado(ws, celda)

        filas_estado = [f for f in filas if f.get("estado") == estado]
        r = 4
        for fila in filas_estado:
            for j, valor in enumerate(_fila_a_lista(fila), start=1):
                ws.cell(row=r, column=j, value=valor)
            r += 1

        fila_final = max(r - 1, 3)
        if fila_final >= 4:
            tabla = Table(displayName=tablas[i], ref=f"A3:R{fila_final}")
            tabla.tableStyleInfo = _tabla_estilo()
            ws.add_table(tabla)

        ws.sheet_state = "hidden"  # igual que en la plantilla original
        _autoancho(ws, {
            "A": 18, "B": 14, "C": 12, "D": 10, "E": 10, "F": 10, "G": 10, "H": 10,
            "I": 14, "J": 10, "K": 10, "L": 10, "M": 12, "N": 10, "O": 12, "P": 14,
            "Q": 10, "R": 20,
        })


def _hoja_estado_calidad(wb, total_filas):
    ws = wb.create_sheet("Estado de calidad")
    ws["B3"] = "Estado de calidad"
    ws["C3"] = "#"
    _estilo_encabezado(ws, "B3")
    _estilo_encabezado(ws, "C3")

    estados = ["Buena", "Aceptable", "Deficiente", "Comprometida"]
    for i, estado in enumerate(estados):
        r = 4 + i
        ws.cell(row=r, column=2, value=estado)
        ws.cell(row=r, column=3, value=f'=COUNTIF(Tabla1[Estado calidad],"{estado}")')

    ws["B8"] = "Total general"
    ws["B8"].font = Font(bold=True)
    ws["C8"] = "=SUM(C4:C7)"
    ws["C8"].font = Font(bold=True)

    _autoancho(ws, {"B": 20, "C": 10})
    # Nota: a propósito NO se agrega ningún gráfico aquí. Los datos (esta hoja,
    # "Potencia regional bandas", "Biomarcadores", etc.) quedan listos para que
    # el usuario arme sus propios gráficos manualmente en Excel si lo desea.
    return ws


def _hoja_potencia_regional_bandas(wb):
    ws = wb.create_sheet("Potencia regional bandas")
    encabezados = ["Región", "Delta (%)", "Theta (%)", "Alfa (%)", "Beta (%)"]
    for idx, encabezado in enumerate(encabezados, start=1):
        celda = f"{get_column_letter(idx + 1)}3"  # arranca en B3
        ws[celda] = encabezado
        _estilo_encabezado(ws, celda)

    columnas_banda = {"Delta (%)": "C", "Theta (%)": "D", "Alfa (%)": "E", "Beta (%)": "F"}

    for i, region in enumerate(REGIONES):
        r = 4 + i
        ws.cell(row=r, column=2, value=region)
        for col_letra, col_tabla in zip(["C", "D", "E", "F"], columnas_banda.keys()):
            ws[f"{col_letra}{r}"] = (
                f"=AVERAGEIFS(Tabla2[{col_tabla}], Tabla2[Región], "
                f"TablaBandasRegion[[#This Row],[Región]])"
            )

    ws["B9"] = "Promedio global"
    ws["B9"].font = Font(bold=True)
    for col_letra in ["C", "D", "E", "F"]:
        ws[f"{col_letra}9"] = f"=SUBTOTAL(101,{col_letra}4:{col_letra}8)"
        ws[f"{col_letra}9"].font = Font(bold=True)

    tabla = Table(displayName="TablaBandasRegion", ref="B3:F9")
    tabla.tableStyleInfo = _tabla_estilo()
    ws.add_table(tabla)
    _autoancho(ws, {"B": 16, "C": 10, "D": 10, "E": 10, "F": 10})
    return ws


def _hoja_biomarcadores(wb):
    ws = wb.create_sheet("Biomarcadores")
    ws["B2"] = "PROMEDIO PROPORCION ENTRE BANDAS (BIOMARCADORES)"
    ws["B2"].font = Font(bold=True)

    encabezados = ["Theta/Alfa", "Delta/Alfa", "Theta/Beta", "Lentificación"]
    columnas_tabla1 = ["Theta/Alfa", "Delta/Alfa", "Theta/Beta", "Lentificación"]
    for idx, encabezado in enumerate(encabezados):
        col = get_column_letter(idx + 2)  # B..E
        celda = f"{col}4"
        ws[celda] = encabezado
        _estilo_encabezado(ws, celda)
        ws[f"{col}5"] = f"=AVERAGE(Tabla1[{columnas_tabla1[idx]}])"

    _autoancho(ws, {"B": 12, "C": 12, "D": 12, "E": 14})
    return ws


def _hoja_calidad_senal(wb):
    ws = wb.create_sheet("Calidad Señal")
    ws["B2"] = "CALIDAD DE LA SEÑAL"
    ws["B2"].font = Font(bold=True)

    encabezados = ["Atípicos (%)", "Sospechosos (%)", "Índice técnico (%)", "SNR (dB)"]
    for idx, encabezado in enumerate(encabezados):
        col = get_column_letter(idx + 2)  # B..E
        celda = f"{col}4"
        ws[celda] = encabezado
        _estilo_encabezado(ws, celda)
        ws[f"{col}5"] = f"=AVERAGE(Tabla1[{encabezado}])"

    _autoancho(ws, {"B": 12, "C": 14, "D": 16, "E": 10})
    return ws


def _hoja_banda_dominante(wb):
    ws = wb.create_sheet("Banda dominante")
    ws["B2"] = "BANDA DOMINANTE"
    ws["B2"].font = Font(bold=True)

    ws["B4"] = "Banda"
    ws["C4"] = "#"
    ws["D4"] = "%"
    for c in ("B4", "C4", "D4"):
        _estilo_encabezado(ws, c)

    for i, banda in enumerate(BANDAS):
        r = 5 + i
        ws.cell(row=r, column=2, value=banda)
        ws.cell(row=r, column=3, value=f'=COUNTIF(Tabla1[Banda dominante],"{banda}")')
        ws[f"D{r}"] = f"=C{r}/$C$9"
        ws[f"D{r}"].number_format = "0%"

    ws["B9"] = "Total"
    ws["B9"].font = Font(bold=True)
    ws["C9"] = "=SUM(C5:C8)"
    ws["C9"].font = Font(bold=True)
    ws["D9"] = "=SUM(D5:D8)"
    ws["D9"].number_format = "0%"
    ws["D9"].font = Font(bold=True)

    _autoancho(ws, {"B": 12, "C": 8, "D": 8})
    return ws