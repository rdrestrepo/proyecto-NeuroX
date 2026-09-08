## =========================================================
## NeuroX - Interfaz EEG con mouse para rotar/arrastrar
## =========================================================

## Imports y dependencias
import os
import sys
import threading
import subprocess
import textwrap
import json
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
import math
import re
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.signal import spectrogram
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageTk

## Configuración de Matplotlib
# Intenta forzar TkAgg SOLO si no hay Qt ya corriendo.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


## Pipeline de procesamiento EEG
from pipeline_eeg_completo import (
    procesar_archivo,
    generar_informe_desde_cache,
    leer_dap,
    limpiar_nombre_canal,
    obtener_nombres_canales_64,
    buscar_logo_neurox,
)

## Generadores de consolidados (nuevo)
from consolidado_excel import generar_consolidado_excel
from consolidado_pdf import combinar_informes_pdf
# -*- coding: utf-8 -*-

AZUL_NEUROX = "#061A5B"
AZUL_MEDIO = "#1D4ED8"
NARANJA_NEUROX = "#FF6A00"
FONDO_APP = "#FFFFFF"
FONDO_PANEL = "#FFFFFF"
FONDO_PANEL_SUAVE = "#F5F5F5"
TEXTO_PRINCIPAL = "#111827"
TEXTO_SECUNDARIO = "#4B5563"
BORDE_SUAVE = "#D9E1EC"
FUENTE_UI = "Segoe UI"
FUENTE_UI_FALLBACK = "Arial"


## Utilidades de DPI / pantalla
def habilitar_alta_resolucion_windows():
    """Evita que Windows escale la ventana como bitmap borroso."""
    if os.name != "nt":
        return

    try:
        import ctypes

        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass

        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def aplicar_escala_tk(raiz):
    """Ajusta la escala de fuentes/widgets al DPI real del monitor."""
    try:
        escala = raiz.winfo_fpixels("1i") / 72.0
        if escala > 0:
            raiz.tk.call("tk", "scaling", escala)
    except Exception:
        pass


## Utilidades de archivos y PDF
def abrir_pdf_en_windows(ruta_pdf: str):
    ruta_pdf = os.path.abspath(ruta_pdf)
    try:
        os.startfile(ruta_pdf)
    except Exception:
        # fallback: abrir con explorer
        subprocess.Popen(["explorer", ruta_pdf], shell=False)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

## Configuración persistente de usuario
def obtener_ruta_config():
    carpeta_usuario = os.path.join(os.path.expanduser("~"), "NeuroX")
    os.makedirs(carpeta_usuario, exist_ok=True)
    return os.path.join(carpeta_usuario, "config_neurox.json")


class _TabBar(tk.Frame):
    """
    Barra de tabs con esquinas superiores redondeadas, estilo Chrome.
    Parámetros:
      fill_width  – True: los tabs se reparten todo el ancho disponible.
      active_fill – color de fondo del tab activo.
      active_fg   – color de texto del tab activo.
    """
    H      = 36   # altura total del widget
    RADIO  = 9    # radio de las esquinas superiores
    PAD_X  = 18   # padding horizontal dentro de cada tab (solo si fill_width=False)
    GAP    = 3    # separación entre tabs

    C_INACT_BG   = "#E4EAF4"
    C_INACT_FG   = TEXTO_SECUNDARIO
    C_BORDE      = BORDE_SUAVE
    C_BASE       = BORDE_SUAVE

    def __init__(self, parent, labels, on_select, fuente=FUENTE_UI,
                 fill_width=False, active_fill=FONDO_PANEL, active_fg=AZUL_NEUROX,
                 **kw):
        kw.setdefault("bg", FONDO_APP)
        kw.setdefault("height", self.H)
        super().__init__(parent, **kw)
        self.pack_propagate(False)

        self._labels      = list(labels)
        self._on_select   = on_select
        self._activo      = 0
        self._rects       = []
        self._fill_width  = fill_width
        self._active_fill = active_fill
        self._active_fg   = active_fg
        self._font_n      = tkfont.Font(family=fuente, size=9)
        self._font_b      = tkfont.Font(family=fuente, size=9, weight="bold")

        self._cv = tk.Canvas(self, bg=FONDO_APP, bd=0, highlightthickness=0,
                             height=self.H)
        self._cv.pack(fill=tk.BOTH, expand=True)
        self._cv.bind("<Configure>", lambda _e: self._draw())
        self._cv.bind("<Button-1>", self._click)

    # ── API pública ───────────────────────────────────────────────────
    def seleccionar(self, idx, dibujar=True):
        self._activo = max(0, min(idx, len(self._labels) - 1))
        if dibujar:
            self._draw()

    # ── Dibujo ───────────────────────────────────────────────────────
    def _draw(self):
        cv = self._cv
        cv.delete("all")
        W  = cv.winfo_width()
        H  = self.H
        if W < 2:
            return

        R   = self.RADIO
        gap = self.GAP
        n   = len(self._labels)
        h_i = H - 5   # altura de tabs inactivos (un poco más bajos)

        # Calcular anchos
        if self._fill_width:
            total_gap = gap * (n - 1)
            tab_w = max(40, (W - total_gap) // n)
            widths = [tab_w] * n
        else:
            widths = [self._font_b.measure(lbl) + self.PAD_X * 2 for lbl in self._labels]

        # Posiciones
        xs = []
        x  = 0 if self._fill_width else 4
        for w in widths:
            xs.append(x)
            x += w + gap

        self._rects = [(xs[i], xs[i] + widths[i]) for i in range(n)]

        # Línea base
        cv.create_line(0, H, W, H, fill=self.C_BASE, width=1)

        # Inactivos primero
        for i, (lbl, xi, wi) in enumerate(zip(self._labels, xs, widths)):
            if i == self._activo:
                continue
            yi = 5
            self._tab(cv, xi, yi, wi, h_i, R,
                      fill=self.C_INACT_BG, borde=self.C_BORDE)
            cv.create_text(xi + wi // 2, yi + h_i // 2,
                           text=lbl, fill=self.C_INACT_FG,
                           font=self._font_n, anchor="center")

        # Activo encima (altura completa, sin borde inferior)
        i  = self._activo
        xi = xs[i]
        wi = widths[i]
        self._tab(cv, xi, 0, wi, H, R,
                  fill=self._active_fill, borde=self.C_BORDE)
        cv.create_rectangle(xi + 1, H - 1, xi + wi - 1, H + 2,
                             fill=self._active_fill, outline="")
        cv.create_text(xi + wi // 2, H // 2,
                       text=self._labels[i], fill=self._active_fg,
                       font=self._font_b, anchor="center")

    def _tab(self, cv, x, y, w, h, r, fill, borde):
        """Dibuja un tab con esquinas superiores redondeadas."""
        # Relleno principal
        cv.create_polygon(
            x,         y + r,
            x + r,     y,
            x + w - r, y,
            x + w,     y + r,
            x + w,     y + h,
            x,         y + h,
            fill=fill, outline="", smooth=False
        )
        # Relleno de los arcos en esquinas (chord elimina triángulo residual)
        for ax, ay in [(x, y), (x + w - 2*r, y)]:
            cv.create_arc(ax, ay, ax + 2*r, ay + 2*r,
                          start=(90 if ax == x else 0),
                          extent=90, fill=fill, outline="", style="chord")
        # Bordes (sin borde inferior)
        cv.create_line(x, y + h, x, y + r, fill=borde)
        cv.create_arc(x, y, x + 2*r, y + 2*r,
                      start=90, extent=90, outline=borde, fill="", style="arc")
        cv.create_line(x + r, y, x + w - r, y, fill=borde)
        cv.create_arc(x + w - 2*r, y, x + w, y + 2*r,
                      start=0, extent=90, outline=borde, fill="", style="arc")
        cv.create_line(x + w, y + r, x + w, y + h, fill=borde)

    # ── Interacción ───────────────────────────────────────────────────
    def _click(self, evt):
        for i, (x0, x1) in enumerate(self._rects):
            if x0 <= evt.x <= x1 and 0 <= evt.y <= self.H:
                if i != self._activo:
                    self._activo = i
                    self._draw()
                    self._on_select(i)
                break


class _LoadingOverlay:
    """GIF animado centrado sobre un widget padre usando place()."""

    def __init__(self, master, ruta_gif, size=80):
        self._master  = master
        self._frames  = []
        self._idx     = 0
        self._job     = None
        self._visible = False

        try:
            gif = Image.open(ruta_gif)
            from PIL import ImageSequence
            for frame in ImageSequence.Iterator(gif):
                img = frame.copy().convert("RGBA").resize((size, size), Image.LANCZOS)
                self._frames.append(ImageTk.PhotoImage(img))
        except Exception:
            pass  # sin frames → show/hide no hace nada

        self._lbl = tk.Label(master, bg=FONDO_PANEL, bd=0, relief="flat")

    def show(self):
        if not self._frames or self._visible:
            return
        self._visible = True
        self._lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._lbl.lift()
        self._animar()

    def hide(self):
        if not self._visible:
            return
        self._visible = False
        if self._job:
            try:
                self._master.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        self._lbl.place_forget()

    def _animar(self):
        if not self._visible or not self._frames:
            return
        self._lbl.config(image=self._frames[self._idx])
        self._idx = (self._idx + 1) % len(self._frames)
        self._job = self._master.after(60, self._animar)


## Aplicación principal
class AppEEG:
    def __init__(self, raiz):
        ## Inicialización de ventana
        self.raiz = raiz
        self.dpi_pantalla = max(96, int(round(self.raiz.winfo_fpixels("1i"))))
        self.raiz.title("NeuroX")
        try:
            icono_ico = resource_path("assets/logo_neurox.ico")
            icono_png = resource_path("assets/logo_neurox.png")

            if os.path.exists(icono_ico):
                self.raiz.iconbitmap(icono_ico)

            if os.path.exists(icono_png):
                self.icono_neurox_tk = tk.PhotoImage(file=icono_png)
                self.raiz.iconphoto(True, self.icono_neurox_tk)

        except Exception:
            pass
        self.raiz.geometry("1280x760")
        self.raiz.configure(bg=FONDO_APP)
        self.fuente_ui = self._resolver_fuente_ui()
        self.configurar_estilos_neurox()

        ## Estado general
        self._cerrando = False
        self.raiz.protocol("WM_DELETE_WINDOW", self._on_cerrar)

        ## Detección de resize activo (para no mostrar GIF ni redibujar durante drag)
        self._resizing       = False
        self._resize_job     = None
        self.raiz.bind("<Configure>", self._on_root_configure, add="+")

        self.carpeta_dcl = ""
        self.archivos_dat = []
        self.cache_ultimo = None
        self.nombre_ultimo = None
        self.ruta_pdf_ultimo = None
        self.canal_activo_simple = None   
        self.hay_grafico_activo = False
        
        self.canal_analisis_simple = None  
        self.var_atipicos = tk.StringVar(value="Canales atípicos: --")
        self.var_sospechosos = tk.StringVar(value="Canales sospechosos: --")
        ## Temporizadores de actualización
        self._after_autograf = None
        self._after_mapa_regional = None
        self._after_ocultar_progreso = None

        ## Estado del análisis / topomaps
        self.banda_heatmap = "Alfa"     # banda activa en el mapa de calor
        self.var_etiquetas_topo = tk.BooleanVar(value=True)

        self.fig_topo = None
        self.cbar_topo = {}
        self._ax_to_band = {}
        self.canvas_topo = None
        self._after_topo_resize = None
        self._topo_redraw_en_progreso = False
        self._ultimo_size_topo = (0, 0)
       
        self._topo_xy = {}

        self.fig_welch = None
        self.canvas_welch = None
        self.widget_welch = None
        self._after_welch_resize = None
        self._welch_redraw_en_progreso = False
        self._pot_rel_welch = None
        self._resumen_welch = None
        self._welch_topo_xy = {}
        self._ax_welch_to_band = {}
        self.cbar_welch = {}
        self.canal_potencia_welch_simple = None

        self.fig_ratios = None
        self.canvas_ratios = None
        self.widget_ratios = None
        self._after_ratios_resize = None
        self._ratios_redraw_en_progreso = False
        self._ratios_principales = None
        self._resumen_ratios = None
        self._ratio_personalizado = None
        self._ratios_topo_xy = {}
        self._ax_ratios_to_nombre = {}
        self.cbar_ratios = {}
        self.canal_ratios_simple = None
        self._ratio_personalizado_activo = False


        self.fig_calidad = None
        self.canvas_calidad = None
        self.widget_calidad = None
        self._resumen_calidad = None
        self._texto_calidad = None
        
        ## Tooltip (hover)
        self._tt_win = None
        self._tt_lbl = None
        self._tt_last = None

        self._img_mapa_regional_tk = None
        self.logo_neurox_tk = None
        
        ## Estado del visor
        self.var_modo = tk.StringVar(value="Filtrado multicanal")

        ## Bandas seleccionadas
        self.bandas = ["Delta", "Theta", "Alfa", "Beta"]
        self.band_vars = {b: tk.BooleanVar(value=(b == "Alfa")) for b in self.bandas}

        ## Ventana temporal y navegación con mouse
        self._fs = None
        self._tf_total = None
        self._t0 = 0.0
        self._dt = 5.0
        self._sl_t0 = None
        self._sl_dt = None
        self._ax_sl_t0 = None
        self._ax_sl_dt = None
        
        self._drag_tiempo_activo = False
        self._drag_x_inicio = None
        self._drag_t0_inicio = None
        self._drag_ultimo_t0 = None
        self._drag_px_por_seg = None
        
        ## IDs de eventos Matplotlib
        self._cid_press = None
        self._cid_release = None
        self._cid_motion = None
        self._cid_hover_multicanal = None
        self._canal_hover_actual = None
        

        ## Controles del visor: multicanal / canal único / FFT
        self.var_ganancia = tk.DoubleVar(value=1.0)
        self.var_fmax = tk.DoubleVar(value=40.0)
        self.var_ventana_espectrograma = tk.DoubleVar(value=2.0)
        self.var_solapamiento_espectrograma = tk.DoubleVar(value=50.0)
        self._fmt_cbar_pct = FuncFormatter(lambda valor, _pos: f"{valor:.1f}".replace(".", ","))

        ## Estado persistente de las vistas
        self._vista_actual = None   # "multicanal" | "canal" | "bandas" | "fft"
        self._ax_main = None
        self._line_canal = None
        self._lines_multicanal = []
        self._offsets_multicanal = None
        
        self._axes_bandas = []
        self._lines_bandas = []
        self._bandas_actuales = []
        self._idx_actual = None
        self._canal_actual = None
        self._nombre_actual = None
        
        
        ## Iconos PNG para botones
        self._ico_copy   = self._cargar_icono_btn("assets/icon_copy.png",   18)
        self._ico_folder = self._cargar_icono_btn("assets/icon_folder.png",  18)

        ## Layout raíz: columna izquierda + columna derecha (sin barra global superior)
        frame_body = ttk.Frame(raiz, padding=(10, 10, 10, 10), style="NeuroX.TFrame")
        frame_body.pack(fill=tk.BOTH, expand=True)
        
        ## Panel izquierdo con scroll
        cont_izq = ttk.Frame(frame_body, width=320, style="NeuroX.TFrame")
        cont_izq.pack(side=tk.LEFT, fill=tk.Y)
        cont_izq.pack_propagate(False)
        self._cont_izq = cont_izq
        
        self.canvas_izq = tk.Canvas(
            cont_izq,
            bg=FONDO_APP,
            bd=0,
            relief="flat",
            highlightthickness=0
        )
        self.scroll_izq = ttk.Scrollbar(
            cont_izq,
            orient="vertical",
            command=self.canvas_izq.yview,
            style="NeuroX.Vertical.TScrollbar"
        )
        def _auto_scroll_izq(lo, hi):
            if float(lo) <= 0.0 and float(hi) >= 1.0:
                self.scroll_izq.grid_remove()
            else:
                self.scroll_izq.grid()
            self.scroll_izq.set(lo, hi)

        self.canvas_izq.configure(yscrollcommand=_auto_scroll_izq)
        cont_izq.grid_rowconfigure(0, weight=1)
        cont_izq.grid_columnconfigure(0, weight=1)
        self.canvas_izq.grid(row=0, column=0, sticky="nsew")
        self.scroll_izq.grid(row=0, column=1, sticky="ns")
        
        # Frame interno que contiene todo lo de la izquierda.
        self.frame_izq = ttk.Frame(self.canvas_izq, style="NeuroX.TFrame")
        self._win_izq = self.canvas_izq.create_window((0, 0), window=self.frame_izq, anchor="nw")
        
        self.panel_izq_fijo_sup = ttk.Frame(self.frame_izq, style="Sidebar.TFrame")
        self.panel_izq_fijo_sup.pack(fill=tk.X)
        
        self.panel_izq_contextual = ttk.Frame(self.frame_izq, style="Sidebar.TFrame")
        self.panel_izq_contextual.pack(fill=tk.X, pady=(8, 0))
        
        self.panel_izq_grafica = ttk.Frame(self.panel_izq_contextual, style="Sidebar.TFrame")
        self.panel_izq_analisis = ttk.Frame(self.panel_izq_contextual, style="Sidebar.TFrame")
        
        self.panel_izq_fijo_inf = ttk.Frame(self.frame_izq, style="Sidebar.TFrame")
        self.panel_izq_fijo_inf.pack(fill=tk.X, pady=(8, 0))
        self._crear_encabezado_neurox()
        
        def _ajustar_scroll(_evt=None):
            self._actualizar_scrollregion_izquierda()
        
        def _ajustar_ancho(_evt):
            # hace que el frame interno tenga el mismo ancho que el canvas
            self.canvas_izq.itemconfig(self._win_izq, width=max(1, _evt.width))
            self.canvas_izq.coords(self._win_izq, 0, 0)
            self._actualizar_scrollregion_izquierda()
        
        self.frame_izq.bind("<Configure>", _ajustar_scroll)
        self.canvas_izq.bind("<Configure>", _ajustar_ancho)
        
        # Scroll con rueda del mouse
        def _scroll_rueda(event):
            if not self._puede_scroll_panel_izquierdo():
                return "break"

            delta = int(-1 * (event.delta / 120))
            if delta != 0:
                self.canvas_izq.yview_scroll(delta, "units")
                self._actualizar_scrollregion_izquierda()
            return "break"
        
        self.canvas_izq.bind("<Enter>", lambda e: self.canvas_izq.bind_all("<MouseWheel>", _scroll_rueda))
        self.canvas_izq.bind("<Leave>", lambda e: self.canvas_izq.unbind_all("<MouseWheel>"))
        
        cont_der = ttk.Frame(frame_body, style="NeuroX.TFrame")
        cont_der.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._cont_der = cont_der

        ## Barra superior del panel derecho: toggle sidebar + tabs globales
        barra = ttk.Frame(cont_der, style="NeuroX.TFrame")
        barra.pack(side=tk.TOP, fill=tk.X)

        self._sidebar_visible = True
        self._btn_toggle_sidebar = ttk.Button(
            barra,
            text="◀",
            width=2,
            command=self._toggle_sidebar,
            style="Secondary.TButton",
        )
        self._btn_toggle_sidebar.pack(side=tk.LEFT, padx=(0, 6))

        ## Área desplazable del panel derecho con scrollbar propio
        cont_der_scroll = ttk.Frame(cont_der, style="NeuroX.TFrame")
        cont_der_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        cont_der_scroll.grid_rowconfigure(0, weight=1)
        cont_der_scroll.grid_columnconfigure(0, weight=1)

        self._canvas_der = tk.Canvas(
            cont_der_scroll,
            bg=FONDO_APP,
            bd=0,
            relief="flat",
            highlightthickness=0,
        )
        self._scroll_der = ttk.Scrollbar(
            cont_der_scroll,
            orient="vertical",
            command=self._canvas_der.yview,
            style="NeuroX.Vertical.TScrollbar",
        )

        def _auto_scroll_der(lo, hi):
            if float(lo) <= 0.0 and float(hi) >= 1.0:
                self._scroll_der.grid_remove()
            else:
                self._scroll_der.grid()
            self._scroll_der.set(lo, hi)

        self._canvas_der.configure(yscrollcommand=_auto_scroll_der)
        self._canvas_der.grid(row=0, column=0, sticky="nsew")
        self._scroll_der.grid(row=0, column=1, sticky="ns")

        ## Tab bar global redondeada: Gráficas / Análisis
        self.tabbar_global = _TabBar(
            barra,
            ["Gráficas", "Análisis"],
            on_select=lambda i: self._activar_tab_global("grafica" if i == 0 else "analisis"),
            fuente=self.fuente_ui,
            fill_width=True,
            active_fill=NARANJA_NEUROX,
            active_fg="#FFFFFF",
        )
        self.tabbar_global.pack(fill=tk.X)

        ## Widgets en la columna izquierda (creados aquí para estar listos al montar el sidebar)
        self.btn_sel = ttk.Button(
            self.panel_izq_fijo_sup,
            text="Seleccionar carpeta",
            command=self.seleccionar_carpeta,
            style="Secondary.TButton",
        )
        self.btn_proc = ttk.Button(
            self.panel_izq_fijo_sup,
            text="Procesar seleccionados",
            command=self._reprocesar_forzado,
            style="Accent.TButton",
            state="disabled",
        )
        self.btn_pdf = ttk.Button(
            self.panel_izq_fijo_sup,
            text="Vista previa",
            command=self.abrir_informe,
            state="disabled",
            style="Primary.TButton",
        )
        self.btn_consolidado = ttk.Button(
            self.panel_izq_fijo_sup,
            text="Descargar consolidado Excel",
            command=self.abrir_ventana_consolidado,
            style="Secondary.TButton",
        )
        self.btn_consolidado_pdf = ttk.Button(
            self.panel_izq_fijo_sup,
            text="Descargar consolidado PDF",
            command=self.abrir_ventana_consolidado_pdf,
            style="Secondary.TButton",
        )
        self.btn_limpiar = ttk.Button(
            self.panel_izq_fijo_inf,
            text="Limpiar",
            command=self.limpiar_datos,
            style="Danger.TButton",
        )
        ## Alias legacy (código anterior referencia estos nombres)
        self.btn_tab_grafica  = self.tabbar_global
        self.btn_tab_analisis = self.tabbar_global
        self.btn_analisis     = self.tabbar_global
        self.btn_grafica      = self.tabbar_global

        ## Botón seleccionar carpeta (columna izquierda)
        self.btn_sel.pack(fill=tk.X, pady=(0, 4))

        ## Estado + progreso
        self.marco_estado_trabajo = ttk.Frame(
            self.panel_izq_fijo_sup,
            style="Sidebar.TFrame",
            padding=(2, 2, 2, 0)
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.var_estado_trabajo = tk.StringVar(value="")
        self.lbl_estado_trabajo = ttk.Label(
            self.marco_estado_trabajo,
            textvariable=self.var_estado_trabajo,
            anchor="w",
            justify="left",
            wraplength=296,
            style="Sidebar.TLabel"
        )
        self.lbl_estado_trabajo.pack(fill=tk.X)

        self.progreso = ttk.Progressbar(
            self.marco_estado_trabajo,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            style="NeuroX.Horizontal.TProgressbar"
        )
        self.progreso.pack(fill=tk.X, pady=(6, 0))

        ## Carpeta actual: label + botones copiar/abrir
        frame_carpeta = ttk.Frame(self.panel_izq_fijo_sup, style="Sidebar.TFrame")
        frame_carpeta.pack(fill=tk.X, pady=(0, 0))

        self.lbl_carpeta = ttk.Label(
            frame_carpeta,
            text="Carpeta actual:\n(no seleccionada)",
            anchor="w",
            justify="left",
            wraplength=220,
            style="Sidebar.TLabel",
        )
        self.lbl_carpeta.pack(side=tk.LEFT, fill=tk.X, expand=True)

        frame_carpeta_btns = ttk.Frame(frame_carpeta, style="Sidebar.TFrame")
        frame_carpeta_btns.pack(side=tk.RIGHT, anchor="n", padx=(4, 0))

        self.btn_copiar_carpeta = ttk.Button(
            frame_carpeta_btns,
            image=self._ico_copy,
            text="" if self._ico_copy else "⎘",
            compound="center",
            command=self._copiar_ruta_carpeta,
            style="Secondary.TButton",
            width=3,
        )
        self.btn_copiar_carpeta.pack(side=tk.LEFT, padx=(0, 2))

        self.btn_abrir_carpeta = ttk.Button(
            frame_carpeta_btns,
            image=self._ico_folder,
            text="" if self._ico_folder else "📁",
            compound="center",
            command=self._abrir_carpeta_en_explorador,
            style="Secondary.TButton",
            width=3,
        )
        self.btn_abrir_carpeta.pack(side=tk.LEFT)

        ## Lista de archivos .dat
        self.marco_lista = ttk.LabelFrame(
            self.panel_izq_fijo_sup,
            text="Archivos .dat  (Ctrl/Shift + clic para elegir varios)",
            padding=10,
            style="Card.TLabelframe"
        )
        self.marco_lista.pack(fill=tk.BOTH, expand=False, pady=(6, 6))

        self._lista_archivos_expandida = True
        self.var_resumen_lista = tk.StringVar(value="Sin archivos cargados.")

        self.frame_lista_header = ttk.Frame(self.marco_lista, style="Sidebar.TFrame")
        self.frame_lista_header.pack(fill=tk.X, pady=(0, 6))

        self.lbl_resumen_lista = ttk.Label(
            self.frame_lista_header,
            textvariable=self.var_resumen_lista,
            style="Sidebar.TLabel",
            anchor="w",
            justify="left"
        )
        self.lbl_resumen_lista.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_toggle_lista = ttk.Button(
            self.frame_lista_header,
            text="Ocultar",
            width=8,
            command=self._toggle_panel_lista_archivos,
            style="Secondary.TButton"
        )
        self.btn_toggle_lista.pack(side=tk.RIGHT, padx=(8, 0))

        self.frame_lista_body = ttk.Frame(self.marco_lista, style="Sidebar.TFrame")
        self.frame_lista_body.pack(fill=tk.BOTH, expand=True)

        # Estilo Treeview sin líneas de árbol ni header
        self.style.configure(
            "Lista.Treeview",
            background=FONDO_PANEL,
            fieldbackground=FONDO_PANEL,
            borderwidth=0,
            relief="flat",
            rowheight=22,
            indent=0,
        )
        self.style.map(
            "Lista.Treeview",
            background=[("selected", AZUL_MEDIO)],
            foreground=[("selected", "#FFFFFF")],
        )
        self.style.layout("Lista.Treeview", [
            ("Treeview.treearea", {"sticky": "nsew"})
        ])

        self.lista = ttk.Treeview(
            self.frame_lista_body,
            show="tree",
            height=10,
            selectmode="extended",
            style="Lista.Treeview",
        )
        self.lista.column("#0", stretch=True, anchor="w")
        self.lista.tag_configure(
            "procesado",
            font=(self.fuente_ui, 9, "bold"),
            foreground=TEXTO_PRINCIPAL,
        )
        self.lista.tag_configure(
            "noprocesado",
            font=(self.fuente_ui, 9),
            foreground="#A8B4C8",
        )
        self.lista.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(
            self.frame_lista_body,
            orient="vertical",
            command=self.lista.yview,
            style="NeuroX.Vertical.TScrollbar"
        )
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista.config(yscrollcommand=scroll.set)
        self.lista.bind("<<TreeviewSelect>>", self.on_select_dat)

        def _evitar_seleccion_vacia(event):
            iid = self.lista.identify_row(event.y)
            if not iid:
                for sel in self.lista.selection():
                    self.lista.selection_remove(sel)
                return "break"

        self.lista.bind("<Button-1>", _evitar_seleccion_vacia)

        ## Botones de archivo debajo de la lista
        self.btn_proc.pack(fill=tk.X, pady=(8, 2))
        self.btn_pdf.pack(fill=tk.X, pady=(2, 0))
        self.btn_consolidado.pack(fill=tk.X, pady=(2, 0))
        self.btn_consolidado_pdf.pack(fill=tk.X, pady=(2, 0))

        ## Controles del visor
        self.lbl_atipicos_grafica = ttk.Label(
            self.panel_izq_grafica,
            textvariable=self.var_atipicos,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_atipicos_grafica.pack(fill=tk.X, pady=(0, 4))
        
        self.lbl_sospechosos_grafica = ttk.Label(
            self.panel_izq_grafica,
            textvariable=self.var_sospechosos,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_sospechosos_grafica.pack(fill=tk.X, pady=(0, 8))
        self.lbl_atipicos_grafica.pack_forget()
        self.lbl_sospechosos_grafica.pack_forget()
        ## Sección Calidad técnica — LabelFrame con toggle interno (vía _crear_bloque_resumen_calidad)
        self._calidad_wrapper_grafica = ttk.Frame(self.panel_izq_grafica, style="Sidebar.TFrame")
        self._calidad_wrapper_grafica.pack(fill=tk.X, pady=(0, 4))
        self.frm_resumen_calidad_grafica = self._crear_bloque_resumen_calidad(
            self._calidad_wrapper_grafica,
            "grafica"
        )
        self.frm_resumen_calidad_grafica.pack(fill=tk.X)
        self.marco_visor = ttk.LabelFrame(
            self.panel_izq_grafica,
            text="Visor",
            padding=10,
            style="Card.TLabelframe"
        )
        self.marco_visor.pack(fill=tk.X)
        _visor_body = self._hacer_colapsable(self.marco_visor)

        self.var_ventana = tk.DoubleVar(value=5.0)

        self.frame_ventana = ttk.Frame(_visor_body, style="Sidebar.TFrame")
        
        self.lbl_ventana = ttk.Label(self.frame_ventana, text="Ventana de tiempo (s):")
        self.lbl_ventana.pack(anchor="w", pady=(6, 0))
        
        self.spin_ventana = ttk.Spinbox(
            self.frame_ventana,
            from_=5.0,
            to=30.0,
            increment=1.0,
            textvariable=self.var_ventana
        )
        self.spin_ventana.pack(fill=tk.X, pady=4)
        
        self.frame_fft = ttk.Frame(_visor_body, style="Sidebar.TFrame")
        
        self.lbl_fmax = ttk.Label(self.frame_fft, text="FFT hasta (Hz):")
        self.lbl_fmax.pack(anchor="w", pady=(6, 0))
        
        self.spin_fmax = ttk.Spinbox(
            self.frame_fft,
            from_=20.0,
            to=100.0,
            increment=5.0,
            textvariable=self.var_fmax
        )
        self.spin_fmax.pack(fill=tk.X, pady=4)

        self.frame_espectrograma = ttk.LabelFrame(
            _visor_body,
            text="Espectrograma",
            padding=8,
            style="Card.TLabelframe"
        )

        self.lbl_ventana_espectrograma = ttk.Label(self.frame_espectrograma, text="Ventana (s):")
        self.lbl_ventana_espectrograma.pack(anchor="w", pady=(0, 0))

        self.spin_ventana_espectrograma = ttk.Spinbox(
            self.frame_espectrograma,
            from_=0.5,
            to=10.0,
            increment=0.5,
            textvariable=self.var_ventana_espectrograma
        )
        self.spin_ventana_espectrograma.pack(fill=tk.X, pady=4)

        self.lbl_solapamiento_espectrograma = ttk.Label(self.frame_espectrograma, text="Solapamiento (%):")
        self.lbl_solapamiento_espectrograma.pack(anchor="w", pady=(6, 0))

        self.spin_solapamiento_espectrograma = ttk.Spinbox(
            self.frame_espectrograma,
            from_=0.0,
            to=95.0,
            increment=5.0,
            textvariable=self.var_solapamiento_espectrograma
        )
        self.spin_solapamiento_espectrograma.pack(fill=tk.X, pady=4)

        self.lbl_escala_espectrograma = ttk.Label(
            self.frame_espectrograma,
            text="Escala: dB",
            anchor="w",
            justify="left"
        )
        self.lbl_escala_espectrograma.pack(fill=tk.X, pady=(6, 0))

        self.lbl_info_espectrograma = ttk.Label(
            self.frame_espectrograma,
            text=(
                "El espectrograma muestra cómo cambia la potencia de cada "
                "frecuencia a lo largo del tiempo. La escala dB mejora el "
                "contraste visual. No corresponde a la magnitud FFT en µV."
            ),
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_info_espectrograma.pack(fill=tk.X, pady=(6, 0))

        self.frame_canal = ttk.Frame(_visor_body, style="Sidebar.TFrame")
        
       
        self.lbl_canal = ttk.Label(self.frame_canal, text="Canal EEG (64):")
        self.lbl_canal.pack(anchor="w", pady=(6, 0))
        
        self.nombres_canales = self._cargar_canales_64()
        self.var_canal_nombre = tk.StringVar(value=self.nombres_canales[0])
        
        self.combo_canal = ttk.Combobox(
            self.frame_canal,
            textvariable=self.var_canal_nombre,
            state="readonly",
            values=self.nombres_canales
        )
        self.combo_canal.current(0)
        self.combo_canal.pack(fill=tk.X, pady=4)
        self.combo_canal.bind("<<ComboboxSelected>>", self._on_combo_canal_change)
        self.var_canal_nombre.trace_add("write", self.actualizar_nav_canal)
        
        self.frame_canal.pack(fill=tk.X)
        
        self.frame_ganancia = ttk.Frame(_visor_body, style="Sidebar.TFrame")
        self.lbl_ganancia = ttk.Label(self.frame_ganancia, text="Ganancia visor multicanal:")
        self.lbl_ganancia.pack(anchor="w", pady=(6, 0))
        
        self.spin_ganancia = ttk.Spinbox(
            self.frame_ganancia,
            from_=0.2,
            to=10.0,
            increment=0.2,
            textvariable=self.var_ganancia
        )
        self.spin_ganancia.pack(fill=tk.X, pady=4)
        
        ## Actualización automática al modificar spinboxes
        self.var_ganancia.trace_add("write", self._programar_autografico)
        self.var_ventana.trace_add("write", self._programar_autografico)
        self.var_fmax.trace_add("write", self._programar_autografico)
        self.var_ventana_espectrograma.trace_add("write", self._programar_autografico)
        self.var_solapamiento_espectrograma.trace_add("write", self._programar_autografico)

        ## Actualización automática al escribir con teclado
        self.spin_ganancia.bind("<KeyRelease>", self._programar_autografico)
        self.spin_ganancia.bind("<<Increment>>", self._programar_autografico)
        self.spin_ganancia.bind("<<Decrement>>", self._programar_autografico)
        
        self.spin_ventana.bind("<KeyRelease>", self._programar_autografico)
        self.spin_ventana.bind("<<Increment>>", self._programar_autografico)
        self.spin_ventana.bind("<<Decrement>>", self._programar_autografico)
        
        self.spin_fmax.bind("<KeyRelease>", self._programar_autografico)
        self.spin_fmax.bind("<<Increment>>", self._programar_autografico)
        self.spin_fmax.bind("<<Decrement>>", self._programar_autografico)

        self.spin_ventana_espectrograma.bind("<KeyRelease>", self._programar_autografico)
        self.spin_ventana_espectrograma.bind("<<Increment>>", self._programar_autografico)
        self.spin_ventana_espectrograma.bind("<<Decrement>>", self._programar_autografico)

        self.spin_solapamiento_espectrograma.bind("<KeyRelease>", self._programar_autografico)
        self.spin_solapamiento_espectrograma.bind("<<Increment>>", self._programar_autografico)
        self.spin_solapamiento_espectrograma.bind("<<Decrement>>", self._programar_autografico)

        ## Controles de bandas
        self.frame_bandas = ttk.LabelFrame(_visor_body, text="Bandas a mostrar (máx 4)", padding=8)
        self.frame_bandas.pack(fill=tk.X, pady=(8, 0))

        self.frame_bandas.configure(style="Card.TLabelframe", text="Bandas a mostrar (máx 4)")
        grid = ttk.Frame(self.frame_bandas, style="Sidebar.TFrame")
        grid.pack(fill=tk.X)

        for i, b in enumerate(self.bandas):
            chk = ttk.Checkbutton(grid, text=b, variable=self.band_vars[b], command=self._limitar_bandas)
            r = i // 2
            c = i % 2
            chk.grid(row=r, column=c, sticky="w", padx=6, pady=2)
            
            
        self.frame_mapa = ttk.LabelFrame(
            _visor_body,
            text="Mapa regional",
            padding=8,
            style="Card.TLabelframe"
        )
        
        ## Mapa regional pequeño (con toggle)
        _mapa_body = self._hacer_colapsable(self.frame_mapa)
        self.canvas_mapa = tk.Canvas(
            _mapa_body,
            width=300,
            height=280,
            bg=FONDO_PANEL,
            highlightthickness=1,
            highlightbackground=BORDE_SUAVE,
            highlightcolor=BORDE_SUAVE,
            bd=0
        )
        self.canvas_mapa.bind("<Configure>", self._on_configure_mapa_regional)
        self.canvas_mapa.pack(fill=tk.X)

        ttk.Button(
            _mapa_body,
            text="Abrir mapa grande (64)",
            command=self.abrir_mapa_grande,
            style="Secondary.TButton"
        ).pack(fill=tk.X, pady=(6, 0))

        ## Leyenda del mapa regional
        self._dibujar_leyenda_mapa(_mapa_body)
        
        # Primer dibujo cuando Tk ya haya terminado de montar la UI.
        self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
        

        
        self.btn_graficar = ttk.Button(
            _visor_body,
            text="Graficar",
            command=self.graficar,
            style="Accent.TButton"
        )
        self.btn_graficar.pack(fill=tk.X, pady=(10, 0))
        self._marco_visor_visible = True
        self.marco_visor.pack_forget()
        self._marco_visor_visible = False

        ## Log y barra de estado inferior izquierda
        marco_log = ttk.LabelFrame(
            self.panel_izq_fijo_inf,
            text="Estado del procesamiento",
            padding=10,
            style="Card.TLabelframe"
        )
        marco_log.pack(fill=tk.X, expand=False, pady=(10, 0))
        _log_body = self._hacer_colapsable(marco_log, {"fill": "both", "expand": True})

        self.txt_log = tk.Text(
            _log_body,
            height=7,
            wrap="word",
            font=(self.fuente_ui, 9),
            relief="flat",
            borderwidth=0,
            bg=FONDO_PANEL_SUAVE,
            fg=TEXTO_PRINCIPAL,
            cursor="arrow",
            insertwidth=0,
            takefocus=0,
            padx=4,
            pady=4,
            highlightthickness=1,
            highlightbackground=BORDE_SUAVE,
            highlightcolor=BORDE_SUAVE
        )
        self.scroll_log = ttk.Scrollbar(
            _log_body,
            orient="vertical",
            command=self.txt_log.yview,
            style="NeuroX.Vertical.TScrollbar"
        )
        self.txt_log.configure(yscrollcommand=self.scroll_log.set)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log.config(state="disabled")
        self.var_barra = tk.StringVar(value="Selecciona una carpeta para comenzar.")

        self.btn_limpiar.pack(fill=tk.X, pady=(8, 0))

        marco_estado_inf = ttk.Frame(self.panel_izq_fijo_inf, style="Sidebar.TFrame")
        marco_estado_inf.pack(fill=tk.X, pady=(8, 0))
        
        self.lbl_barra = ttk.Label(
            marco_estado_inf,
            textvariable=self.var_barra,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_barra.pack(fill=tk.X)

        ## Panel derecho principal (dentro del canvas desplazable)
        self.panel_derecho = ttk.Frame(self._canvas_der, style="NeuroX.TFrame")
        self._win_der = self._canvas_der.create_window((0, 0), window=self.panel_derecho, anchor="nw")

        def _sincronizar_panel_der(_evt=None):
            w = max(1, self._canvas_der.winfo_width())
            h = max(1, self._canvas_der.winfo_height())
            self._canvas_der.itemconfig(self._win_der, width=w, height=h)
            self._canvas_der.coords(self._win_der, 0, 0)
            self._canvas_der.configure(scrollregion=(0, 0, w, h))

        self.panel_derecho.bind("<Configure>", lambda _e: _sincronizar_panel_der())
        self._canvas_der.bind("<Configure>", lambda _e: _sincronizar_panel_der())
        
        ## Panel de gráfica
        self.panel_grafica = ttk.Frame(self.panel_derecho, style="NeuroX.TFrame")
        self.panel_grafica.pack(fill=tk.BOTH, expand=True)

        self.modos_grafica_tabs = {
            "Multicanal": "Filtrado multicanal",
            "Canal Filtrado": "Canal filtrado",
            "FFT": "FFT filtrada",
            "Bandas": "Bandas",
            "Espectrograma": "Espectrograma",
        }
        self._sincronizando_tab_grafica = False
        self._tabs_grafica_visible = False

        ## Wrapper: tabbar redondeada + notebook headless (sin strip visual)
        self._frame_tabs_grafica = ttk.Frame(self.panel_grafica, style="NeuroX.TFrame")

        self.tabbar_grafica = _TabBar(
            self._frame_tabs_grafica,
            list(self.modos_grafica_tabs.keys()),
            on_select=self._on_tabbar_grafica_click,
            fuente=self.fuente_ui,
            fill_width=False,
            active_fill=AZUL_NEUROX,
            active_fg="#FFFFFF",
        )
        self.tabbar_grafica.pack(fill=tk.X)

        self.tabs_grafica = ttk.Notebook(
            self._frame_tabs_grafica, height=1, style="Headless.TNotebook"
        )
        self.frames_tabs_grafica = {}
        for texto_tab in self.modos_grafica_tabs:
            frame_tab = ttk.Frame(self.tabs_grafica, style="Card.TFrame")
            self.tabs_grafica.add(frame_tab, text=texto_tab)
            self.frames_tabs_grafica[texto_tab] = frame_tab
        self.tabs_grafica.bind("<<NotebookTabChanged>>", self._on_tab_grafica)

        self.frame_nav_canal = ttk.Frame(self.panel_grafica, style="Card.TFrame", padding=(8, 4, 8, 4))
        self.btn_canal_prev = ttk.Button(
            self.frame_nav_canal,
            text="<",
            width=3,
            command=lambda: self.cambiar_canal_grafica(-1),
            style="Secondary.TButton"
        )
        self.btn_canal_prev.pack(side=tk.LEFT)

        self.lbl_titulo_canal_nav = ttk.Label(
            self.frame_nav_canal,
            text="Canal actual: --",
            anchor="center",
            justify="center",
            font=(self.fuente_ui, 11, "bold"),
            foreground=AZUL_NEUROX
        )
        self.lbl_titulo_canal_nav.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.btn_canal_next = ttk.Button(
            self.frame_nav_canal,
            text=">",
            width=3,
            command=lambda: self.cambiar_canal_grafica(1),
            style="Secondary.TButton"
        )
        self.btn_canal_next.pack(side=tk.RIGHT)
        self._nav_canal_visible = False
        
        self.frame_visor_plot = ttk.Frame(self.panel_grafica, style="Card.TFrame", padding=(8, 8, 8, 6))
        self.frame_visor_plot.pack(fill=tk.BOTH, expand=True)
        
        ## Layout interno del visor
        self.frame_visor_plot.rowconfigure(0, weight=1)
        self.frame_visor_plot.rowconfigure(1, weight=0)
        self.frame_visor_plot.columnconfigure(0, weight=1)
        
        self.fig = plt.Figure(figsize=(8.0, 7.8), dpi=self.dpi_pantalla)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_visor_plot)
        
        widget_canvas = self.canvas.get_tk_widget()
        widget_canvas.grid(row=0, column=0, sticky="nsew")
        
        toolbar_frame = ttk.Frame(self.frame_visor_plot, style="Card.TFrame")
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self._activar_pan_tiempo()

        ## Overlay GIF de carga — centrado sobre el área de gráfica
        self._loading = _LoadingOverlay(self.panel_derecho, resource_path("assets/loading.gif"), size=80)

        ## Panel de análisis
        self.panel_analisis = ttk.Frame(self.panel_derecho, style="NeuroX.TFrame")
        self._er_filtrado = None
        self._pos_1010 = {}

        self.tabbar_analisis = _TabBar(
            self.panel_analisis,
            ["Potencia relativa", "Proporción de bandas", "Calidad técnica"],
            on_select=self._on_tabbar_analisis_click,
            fuente=self.fuente_ui,
            fill_width=False,
            active_fill=AZUL_NEUROX,
            active_fg="#FFFFFF",
        )
        self.tabbar_analisis.pack(fill=tk.X)

        self.tabs_analisis = ttk.Notebook(self.panel_analisis, height=1, style="Headless.TNotebook")
        self.tabs_analisis.pack(fill=tk.BOTH, expand=True)

        self.tab_energia_relativa = ttk.Frame(self.tabs_analisis, style="Card.TFrame")
        self.tab_potencia_welch = ttk.Frame(self.tabs_analisis, style="Card.TFrame")
        self.tab_ratios_bandas = ttk.Frame(self.tabs_analisis, style="Card.TFrame")
        self.tab_calidad_senal = ttk.Frame(self.tabs_analisis, style="Card.TFrame")
        self.tabs_analisis.add(self.tab_potencia_welch, text="Potencia relativa")
        self.tabs_analisis.add(self.tab_ratios_bandas, text="Proporción de bandas")
        self.tabs_analisis.add(self.tab_calidad_senal, text="Calidad técnica")
        self.tabs_analisis.bind("<<NotebookTabChanged>>", self._on_tab_analisis)

        self.tab_analisis = self.tab_energia_relativa   # para reutilizar el código actual
        self._crear_tab_analisis()
        self._crear_tab_potencia_welch()
        self._crear_tab_ratios_bandas()
        self._crear_tab_calidad_senal()
        self._crear_panel_izq_analisis()
        
        # Inicia mostrando el panel de gráfica.
        self.panel_analisis.pack_forget()

        
        
        self._refrescar_ui_modo()
        self._mostrar_panel_izquierdo("grafica")
        self._mostrar_placeholder_visor()
        self.cargar_config()

    def _resolver_fuente_ui(self):
        try:
            familias = {str(nombre).lower() for nombre in tkfont.families(self.raiz)}
        except Exception:
            familias = set()

        for nombre in (FUENTE_UI, FUENTE_UI_FALLBACK):
            if str(nombre).lower() in familias:
                return nombre
        return FUENTE_UI_FALLBACK if familias else FUENTE_UI

    def configurar_estilos_neurox(self):
        self.style = ttk.Style(self.raiz)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        fuente_ui = getattr(self, "fuente_ui", FUENTE_UI)
        fuente_base = (fuente_ui, 9)
        fuente_bold = (fuente_ui, 9, "bold")
        fuente_titulo = (fuente_ui, 10, "bold")

        self.style.configure(".", font=fuente_base)
        self.style.configure("TFrame", background=FONDO_PANEL)
        self.style.configure("NeuroX.TFrame", background=FONDO_APP)
        self.style.configure("Sidebar.TFrame", background=FONDO_PANEL)
        self.style.configure("Card.TFrame", background=FONDO_PANEL)

        self.style.configure("TLabel", background=FONDO_PANEL, foreground=TEXTO_PRINCIPAL, font=fuente_base)
        self.style.configure("Sidebar.TLabel", background=FONDO_PANEL, foreground=TEXTO_SECUNDARIO, font=fuente_base)
        self.style.configure("Header.TLabel", background=FONDO_PANEL, foreground=AZUL_NEUROX, font=fuente_titulo)
        self.style.configure("Logo.TLabel", background=FONDO_PANEL, foreground=AZUL_NEUROX, font=(fuente_ui, 19, "bold"))
        self.style.configure("Subtle.TLabel", background=FONDO_PANEL, foreground=TEXTO_SECUNDARIO, font=(fuente_ui, 8))

        self.style.configure("TLabelframe", background=FONDO_PANEL, borderwidth=1, relief="solid",
                             bordercolor=BORDE_SUAVE)
        self.style.configure("TLabelframe.Label", background=FONDO_PANEL, foreground=AZUL_NEUROX, font=fuente_bold)
        self.style.configure("Card.TLabelframe", background=FONDO_PANEL, borderwidth=1, relief="solid",
                             bordercolor=BORDE_SUAVE)
        self.style.configure("Card.TLabelframe.Label", background=FONDO_PANEL, foreground=AZUL_NEUROX, font=fuente_bold)

        self.style.configure(
            "Primary.TButton",
            font=fuente_bold,
            padding=(12, 8),
            background=AZUL_NEUROX,
            foreground="#FFFFFF",
            borderwidth=0
        )
        self.style.map(
            "Primary.TButton",
            background=[
                ("disabled", "#B8C4D6"),
                ("pressed", "#04133D"),
                ("active", "#0A257A"),
            ],
            foreground=[("disabled", "#FFFFFF")]
        )

        self.style.configure(
            "Accent.TButton",
            font=fuente_bold,
            padding=(12, 8),
            background=NARANJA_NEUROX,
            foreground="#FFFFFF",
            borderwidth=0
        )
        self.style.map(
            "Accent.TButton",
            background=[
                ("disabled", "#F7C9A4"),
                ("pressed", "#D85700"),
                ("active", "#FF7F26"),
            ],
            foreground=[("disabled", "#FFFFFF")]
        )

        self.style.configure(
            "Secondary.TButton",
            font=fuente_bold,
            padding=(12, 8),
            background="#E8EEF8",
            foreground=AZUL_NEUROX,
            borderwidth=0
        )
        self.style.map(
            "Secondary.TButton",
            background=[
                ("disabled", "#EEF2F7"),
                ("pressed", "#D5E1F4"),
                ("active", "#DCE8FF"),
            ],
            foreground=[("disabled", "#8191A8")]
        )

        self.style.configure(
            "Danger.TButton",
            font=fuente_bold,
            padding=(12, 8),
            background=FONDO_PANEL,
            foreground=TEXTO_SECUNDARIO,
            borderwidth=1,
            relief="solid",
            bordercolor=BORDE_SUAVE,
        )
        self.style.map(
            "Danger.TButton",
            background=[
                ("disabled", FONDO_PANEL),
                ("pressed", "#B91C1C"),
                ("active", "#DC2626"),
            ],
            foreground=[
                ("disabled", TEXTO_SECUNDARIO),
                ("pressed", "#FFFFFF"),
                ("active", "#FFFFFF"),
            ],
        )

        self.style.configure(
            "NeuroX.Horizontal.TProgressbar",
            troughcolor=FONDO_PANEL_SUAVE,
            background=AZUL_MEDIO,
            lightcolor=AZUL_MEDIO,
            darkcolor=AZUL_NEUROX,
            bordercolor=BORDE_SUAVE,
            thickness=10
        )
        try:
            self.style.configure(
                "NeuroX.Vertical.TScrollbar",
                gripcount=0,
                background="#CBD5E1",
                darkcolor="#CBD5E1",
                lightcolor="#CBD5E1",
                troughcolor="#F1F5F9",
                bordercolor="#F1F5F9",
                arrowcolor=AZUL_NEUROX,
                relief="flat",
                arrowsize=11,
                width=10
            )
            self.style.configure(
                "NeuroX.Horizontal.TScrollbar",
                background="#CBD5E1",
                darkcolor="#CBD5E1",
                lightcolor="#CBD5E1",
                troughcolor="#F1F5F9",
                bordercolor="#F1F5F9",
                arrowcolor=AZUL_NEUROX,
                relief="flat",
                arrowsize=11
            )
        except Exception:
            pass

        self.style.configure("TNotebook", background=FONDO_APP, borderwidth=0)
        self.style.configure("NeuroX.TNotebook", background=FONDO_APP, borderwidth=0, tabmargins=(0, 0, 0, 0))
        # Notebook sin tab-strip visual (los tabs los maneja _TabBar)
        self.style.configure("Headless.TNotebook", background=FONDO_APP, borderwidth=0)
        try:
            self.style.layout("Headless.TNotebook", [("Notebook.client", {"sticky": "nsew"})])
            self.style.layout("Headless.TNotebook.Tab", [])   # Tab invisible → sin duplicados
        except Exception:
            pass
        self.style.configure(
            "NeuroX.TNotebook.Tab",
            background="#E8EEF8",
            foreground=TEXTO_SECUNDARIO,
            padding=(14, 7),
            font=fuente_bold,
            borderwidth=0,
        )
        self.style.map(
            "NeuroX.TNotebook.Tab",
            background=[
                ("selected", AZUL_NEUROX),
                ("active", "#DCE8FF"),
            ],
            foreground=[
                ("selected", "#FFFFFF"),
                ("active", AZUL_NEUROX),
            ],
            expand=[("selected", (0, 0, 0, 2))],
        )

        self.style.configure(
            "TCombobox",
            padding=4,
            fieldbackground=FONDO_PANEL,
            background=FONDO_PANEL,
            foreground=TEXTO_PRINCIPAL
        )
        self.style.configure(
            "TSpinbox",
            padding=4,
            fieldbackground=FONDO_PANEL,
            background=FONDO_PANEL,
            foreground=TEXTO_PRINCIPAL
        )
        self.style.configure("TCheckbutton", background=FONDO_PANEL, foreground=TEXTO_PRINCIPAL, font=fuente_base)
        self.style.configure(
            "NeuroX.Treeview",
            rowheight=24,
            font=(fuente_ui, 9),
            background=FONDO_PANEL,
            fieldbackground=FONDO_PANEL,
            foreground=TEXTO_PRINCIPAL,
            borderwidth=0,
            relief="flat"
        )
        self.style.map(
            "NeuroX.Treeview",
            background=[("selected", AZUL_MEDIO)],
            foreground=[("selected", "#FFFFFF")]
        )
        self.style.configure(
            "NeuroX.Treeview.Heading",
            font=(fuente_ui, 9, "bold"),
            background="#E8EEF7",
            foreground=AZUL_NEUROX,
            borderwidth=0,
            relief="flat",
            padding=(6, 4)
        )
        self.style.map(
            "NeuroX.Treeview.Heading",
            background=[("active", "#DCE8FF")],
            foreground=[("active", AZUL_NEUROX)]
        )
        self.style.configure("Treeview", rowheight=24, font=(fuente_ui, 8.5))
        self.style.configure("Treeview.Heading", font=(fuente_ui, 8.5, "bold"))

    def _crear_encabezado_neurox(self):
        self.frame_marca = ttk.Frame(self.panel_izq_fijo_sup, style="Sidebar.TFrame", padding=(2, 0, 2, 6))
        self.frame_marca.pack(fill=tk.X)

        self.logo_neurox_tk = None
        logo_path = buscar_logo_neurox()
        if logo_path:
            try:
                with Image.open(logo_path) as imagen:
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                    imagen = imagen.copy()
                    imagen.thumbnail((300, 100), resample)
                    self.logo_neurox_tk = ImageTk.PhotoImage(imagen)
            except Exception:
                self.logo_neurox_tk = None

        if self.logo_neurox_tk is not None:
            ttk.Label(self.frame_marca, image=self.logo_neurox_tk, style="Sidebar.TLabel").pack(anchor="center")
        else:
            ttk.Label(self.frame_marca, text="NeuroX", style="Logo.TLabel").pack(anchor="center")
        return

        self.frame_marca = ttk.Frame(self.panel_izq_fijo_sup, style="Sidebar.TFrame", padding=(2, 0, 2, 6))
        self.frame_marca.pack(fill=tk.X)

        logo_path = buscar_logo_neurox()
        if logo_path:
            try:
                imagen = Image.open(logo_path)
                nombre_logo = os.path.basename(logo_path).lower()
                if "icono" in nombre_logo:
                    ancho_objetivo = 50
                elif "nombre_neurox" in nombre_logo:
                    ancho_objetivo = 190
                else:
                    ancho_objetivo = 220

                ancho_original, alto_original = imagen.size
                if ancho_original > 0 and alto_original > 0:
                    alto_objetivo = max(1, int(round((alto_original / ancho_original) * ancho_objetivo)))
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                    imagen = imagen.resize((ancho_objetivo, alto_objetivo), resample)
                    self.logo_neurox_tk = ImageTk.PhotoImage(imagen)
            except Exception:
                self.logo_neurox_tk = None

        if self.logo_neurox_tk is not None:
            ttk.Label(self.frame_marca, image=self.logo_neurox_tk, style="Sidebar.TLabel").pack(anchor="w")
        else:
            ttk.Label(self.frame_marca, text="NeuroX", style="Logo.TLabel").pack(anchor="w")

        ttk.Label(
            self.frame_marca,
            text="Procesamiento y análisis de señales EEG",
            style="Subtle.TLabel"
        ).pack(anchor="w", pady=(3, 0))

    ## =========================================================
    ## Cierre, log y estado de trabajo
    ## =========================================================
    def _on_cerrar(self):
        self._cerrando = True
        try:
            self.raiz.destroy()
        except Exception:
            pass

    def log(self, msg):
        if self._cerrando:
            return
        try:
            self.txt_log.config(state="normal")
            self.txt_log.insert(tk.END, str(msg) + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state="disabled")
        except Exception:
            pass

    def limpiar_log(self):
        if self._cerrando:
            return
        try:
            self.txt_log.config(state="normal")
            self.txt_log.delete("1.0", tk.END)
            self.txt_log.config(state="disabled")
        except Exception:
            pass

    def _set_barra(self, msg):
        if self._cerrando:
            return
        try:
            self.var_barra.set(msg)
        except Exception:
            pass

    def _actualizar_barra_modo_grafica(self, modo=None, nombre_canal=None):
        modo = self._normalizar_modo_grafica(modo or self.var_modo.get())
        nombre = self._nombre_simple(nombre_canal) if nombre_canal else ""

        if modo == "Filtrado multicanal":
            self._set_barra("Señal EEG multicanal mostrada.")
        elif modo == "Canal filtrado":
            self._set_barra(
                f"Canal filtrado mostrado para {nombre}." if nombre else "Canal filtrado mostrado."
            )
        elif modo == "FFT filtrada":
            self._set_barra(
                f"FFT filtrada mostrada para {nombre}." if nombre else "FFT filtrada mostrada."
            )
        elif modo == "Bandas":
            self._set_barra(
                f"Bandas mostradas para {nombre}." if nombre else "Bandas mostradas."
            )
        elif modo == "Espectrograma":
            self._set_barra(
                f"Espectrograma mostrado para {nombre}." if nombre else "Espectrograma mostrado."
            )

    def _actualizar_barra_modo_analisis(self, tab=None):
        tab = tab or self._tab_analisis_actual()

        if tab == "welch":
            self._set_barra("Análisis: potencia relativa.")
        elif tab == "ratios":
            self._set_barra("Análisis: proporción de bandas.")
        elif tab == "calidad":
            self._set_barra("Análisis: calidad técnica.")

    def guardar_config(self):
        try:
            config = {
                "carpeta_dcl": self.carpeta_dcl
            }

            ruta_config = obtener_ruta_config()
            with open(ruta_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            self.log(f"Configuración guardada: {ruta_config}")
        except Exception as e:
            try:
                self.log(f"No se pudo guardar configuración: {e}")
            except Exception:
                pass

    def cargar_config(self):
        try:
            ruta_config = obtener_ruta_config()

            if not os.path.exists(ruta_config):
                return

            with open(ruta_config, "r", encoding="utf-8") as f:
                config = json.load(f)

            carpeta = config.get("carpeta_dcl", "")

            if not carpeta or not os.path.isdir(carpeta):
                self._set_barra("La última carpeta guardada ya no existe. Selecciona una carpeta.")
                return

            self.carpeta_dcl = carpeta
            self.lbl_carpeta.config(
                text=f"Carpeta actual:\n{self._formatear_ruta_panel(self.carpeta_dcl)}"
            )

            archivos = [f for f in os.listdir(self.carpeta_dcl) if f.lower().endswith(".dat")]
            archivos.sort()
            self.archivos_dat = archivos

            self.lista.delete(*self.lista.get_children())
            for f in archivos:
                self.lista.insert("", tk.END, iid=f, text=f, tags=("noprocesado",))

            if archivos:
                self.var_resumen_lista.set(f"{len(archivos)} archivo(s) .dat")
                self._set_barra(f"Carpeta restaurada: {len(archivos)} archivo(s) .dat encontrado(s).")
                self.log(f"Carpeta restaurada desde configuración: {self.carpeta_dcl}")
            else:
                self.var_resumen_lista.set("0 archivos .dat")
                self._set_barra("Carpeta restaurada, pero no contiene archivos .dat.")
                self.log("Carpeta restaurada, pero no contiene archivos .dat.")
            self.raiz.after_idle(self._actualizar_colores_lista)

        except Exception as e:
            try:
                self.log(f"No se pudo cargar configuración: {e}")
            except Exception:
                pass

    def mostrar_progreso(self):
        if self._cerrando:
            return
        if self._after_ocultar_progreso is not None:
            try:
                self.raiz.after_cancel(self._after_ocultar_progreso)
            except Exception:
                pass
            self._after_ocultar_progreso = None
        if not self.marco_estado_trabajo.winfo_ismapped():
            self.marco_estado_trabajo.pack(fill=tk.X, pady=(0, 6), before=self.lbl_carpeta)
        if not self.progreso.winfo_ismapped():
            self.progreso.pack(fill=tk.X, pady=(6, 0))

    def ocultar_progreso(self):
        if self._cerrando:
            return
        self._after_ocultar_progreso = None
        self.progress_var.set(0.0)
        self.var_estado_trabajo.set("")
        try:
            self.marco_estado_trabajo.pack_forget()
        except Exception:
            pass

    def set_progreso(self, valor, mensaje=None):
        if self._cerrando:
            return
        try:
            valor = max(0.0, min(100.0, float(valor)))
        except Exception:
            valor = 0.0

        self.mostrar_progreso()

        self.progress_var.set(valor)
        self.var_estado_trabajo.set(f"{mensaje or 'Procesando...'} ({valor:.0f}%)")

        try:
            self.raiz.update_idletasks()
        except Exception:
            pass

    def reset_progreso(self, mensaje="Listo"):
        if self._cerrando:
            return
        self.progress_var.set(0.0)
        self.var_estado_trabajo.set(str(mensaje or ""))
        try:
            self.raiz.update_idletasks()
        except Exception:
            pass

    def finalizar_progreso(self, mensaje="Procesamiento finalizado.", valor=100.0, delay_ms=1800):
        if self._cerrando:
            return
        self.set_progreso(valor, mensaje)
        self._terminar_trabajo()
        self._after_ocultar_progreso = self.raiz.after(delay_ms, self.ocultar_progreso)

    def _toggle_panel_lista_archivos(self):
        if not hasattr(self, "frame_lista_body"):
            return

        self._lista_archivos_expandida = not getattr(self, "_lista_archivos_expandida", True)
        if self._lista_archivos_expandida:
            self.frame_lista_body.pack(fill=tk.BOTH, expand=True)
            self.btn_toggle_lista.config(text="Ocultar")
        else:
            self.frame_lista_body.pack_forget()
            self.btn_toggle_lista.config(text="Mostrar")

        self._actualizar_scrollregion_izquierda()

    def _aplicar_estilo_treeview(self, tree):
        if tree is None:
            return
        try:
            tree.tag_configure("odd", background="#FFFFFF", foreground=TEXTO_PRINCIPAL)
            tree.tag_configure("even", background=FONDO_PANEL_SUAVE, foreground=TEXTO_PRINCIPAL)
        except Exception:
            pass

    def _insertar_fila_treeview(self, tree, values, iid=None):
        if tree is None:
            return

        indice = len(tree.get_children())
        tag = "odd" if (indice % 2) == 0 else "even"
        kwargs = {
            "values": tuple(values),
            "tags": (tag,),
        }
        if iid is not None:
            kwargs["iid"] = iid
        tree.insert("", tk.END, **kwargs)

    def _color_estado_calidad(self, estado):
        colores = {
            "Buena": "#15803D",
            "Aceptable": "#CA8A04",
            "Comprometida": "#EA580C",
            "Deficiente": "#DC2626",
        }
        return colores.get(str(estado or "").strip(), "#1F2937")

    def _metricas_calidad_resumen(self, resumen):
        if not isinstance(resumen, dict):
            return {
                "pct_ponderado": 0.0,
                "pct_atipicos": 0.0,
                "pct_sospechosos": 0.0,
                "snr_db": 0.0,
                "region": "No determinada",
                "estado": "--",
            }

        try:
            pct_ponderado = float(
                resumen.get("porcentaje_ponderado", resumen.get("porcentaje_afectado", 0.0))
            )
        except Exception:
            pct_ponderado = 0.0

        try:
            pct_atipicos = float(resumen.get("porcentaje_atipicos", 0.0))
        except Exception:
            pct_atipicos = 0.0

        try:
            pct_sospechosos = float(resumen.get("porcentaje_sospechosos", 0.0))
        except Exception:
            pct_sospechosos = 0.0

        try:
            snr_db = float(resumen.get("snr", {}).get("snr_global_db", 0.0))
        except Exception:
            snr_db = 0.0

        return {
            "pct_ponderado": pct_ponderado,
            "pct_atipicos": pct_atipicos,
            "pct_sospechosos": pct_sospechosos,
            "snr_db": snr_db,
            "region": resumen.get("analisis_regional", {}).get("region_predominante", "No determinada"),
            "estado": resumen.get("estado_global", "--"),
        }

    def _formatear_lista_canales_compacta(self, canales, limite=8):
        if not canales:
            return "ninguno"
        limpios = [self._nombre_simple(c) or str(c).strip() for c in canales]
        limpios = [c for c in limpios if c]
        if not limpios:
            return "ninguno"
        return ", ".join(dict.fromkeys(limpios))

    def _on_combo_canal_change(self, _evt=None):
        self.actualizar_nav_canal()
        nombre = self.var_canal_nombre.get()
        self.canal_activo_simple = self._nombre_simple(nombre)
        self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
        if self.cache_ultimo:
            self.graficar()

    def _indice_canal_actual(self):
        if not getattr(self, "nombres_canales", None):
            return None

        try:
            idx = int(self.combo_canal.current())
        except Exception:
            idx = -1

        if 0 <= idx < len(self.nombres_canales):
            return idx

        actual = self._nombre_simple(self.var_canal_nombre.get())
        for i, nombre in enumerate(self.nombres_canales):
            if self._nombre_simple(nombre) == actual:
                return i
        return 0

    def _mostrar_nav_canal(self, visible):
        if not hasattr(self, "frame_nav_canal"):
            return

        visible = bool(visible)
        if visible == getattr(self, "_nav_canal_visible", False):
            return

        if visible:
            self.frame_nav_canal.pack(fill=tk.X, pady=(6, 6), before=self.frame_visor_plot)
        else:
            self.frame_nav_canal.pack_forget()

        self._nav_canal_visible = visible

    def actualizar_nav_canal(self, *_args):
        if not hasattr(self, "lbl_titulo_canal_nav"):
            return

        modo = self._normalizar_modo_grafica(self.var_modo.get())
        mostrar = bool(self.cache_ultimo) and modo in ("Canal filtrado", "FFT filtrada", "Bandas", "Espectrograma")
        self._mostrar_nav_canal(mostrar)

        if not mostrar:
            return

        idx = self._indice_canal_actual()
        if idx is None or not self.nombres_canales:
            self.lbl_titulo_canal_nav.config(text="Canal actual: --")
            self.btn_canal_prev.config(state="disabled")
            self.btn_canal_next.config(state="disabled")
            return

        nombre = self._nombre_simple(self.nombres_canales[idx]) or self.nombres_canales[idx]
        self.lbl_titulo_canal_nav.config(text=f"Canal actual: {nombre} (#{idx + 1})")
        estado_botones = "normal" if len(self.nombres_canales) > 1 else "disabled"
        self.btn_canal_prev.config(state=estado_botones)
        self.btn_canal_next.config(state=estado_botones)

    def cambiar_canal_grafica(self, direccion):
        modo = self._normalizar_modo_grafica(self.var_modo.get())
        if modo == "Filtrado multicanal" or not self.nombres_canales:
            self.actualizar_nav_canal()
            return

        idx_actual = self._indice_canal_actual()
        if idx_actual is None:
            return

        try:
            direccion = int(direccion)
        except Exception:
            direccion = 0

        nuevo_idx = (idx_actual + direccion) % len(self.nombres_canales)
        self.combo_canal.current(nuevo_idx)
        self.var_canal_nombre.set(self.nombres_canales[nuevo_idx])
        self.canal_activo_simple = self._nombre_simple(self.nombres_canales[nuevo_idx])
        self.actualizar_nav_canal()

        if self.cache_ultimo:
            self.graficar()
        self.raiz.after_idle(self._dibujar_mapa_regional_resumen)

    def _crear_bloque_resumen_calidad(self, parent, prefijo):
        _fb = (self.fuente_ui, 9, "bold")
        _fn = (self.fuente_ui, 9)

        frame = ttk.LabelFrame(parent, text="Calidad técnica", padding=10, style="Card.TLabelframe")
        labels = {}

        # Cuerpo colapsable (igual que Visor)
        body = self._hacer_colapsable(frame)

        labels["estado"] = ttk.Label(
            body,
            text="Estado global: --",
            font=(self.fuente_ui, 10, "bold"),
            anchor="w",
            justify="left",
            wraplength=280,
            foreground=self._color_estado_calidad(None),
        )
        labels["estado"].pack(fill=tk.X, pady=(0, 6))

        ttk.Label(body, text="Índice técnico ponderado:", font=_fb, anchor="w").pack(fill=tk.X)
        labels["pct"] = ttk.Label(body, text="--", font=_fn, anchor="w", wraplength=280)
        labels["pct"].pack(fill=tk.X)

        labels["pct_detalle"] = ttk.Label(
            body, text="Atípicos: -- | Sospechosos: --",
            anchor="w", justify="left", wraplength=280, style="Sidebar.TLabel",
        )
        labels["pct_detalle"].pack(fill=tk.X, pady=(0, 4))

        ttk.Label(body, text="SNR global:", font=_fb, anchor="w").pack(fill=tk.X)
        labels["snr"] = ttk.Label(body, text="--", font=_fn, anchor="w", wraplength=280)
        labels["snr"].pack(fill=tk.X)

        ttk.Label(body, text="Región predominante:", font=_fb, anchor="w").pack(fill=tk.X, pady=(4, 0))
        labels["region"] = ttk.Label(body, text="--", font=_fn, anchor="w", wraplength=280)
        labels["region"].pack(fill=tk.X)

        ttk.Label(body, text="Canales atípicos:", font=_fb, anchor="w").pack(fill=tk.X, pady=(6, 0))
        labels["atipicos"] = ttk.Label(body, text="--", font=_fn, anchor="w", wraplength=280, justify="left")
        labels["atipicos"].pack(fill=tk.X)

        ttk.Label(body, text="Canales sospechosos:", font=_fb, anchor="w").pack(fill=tk.X, pady=(4, 0))
        labels["sospechosos"] = ttk.Label(body, text="--", font=_fn, anchor="w", wraplength=280, justify="left")
        labels["sospechosos"].pack(fill=tk.X)

        setattr(self, f"labels_resumen_calidad_{prefijo}", labels)
        return frame

    def _actualizar_bloques_resumen_calidad(self):
        resumen = self.cargar_resumen_calidad(mostrar_warning=False)

        for prefijo in ("grafica", "analisis"):
            labels = getattr(self, f"labels_resumen_calidad_{prefijo}", None)
            if not labels:
                continue

            if resumen is None:
                labels["estado"].config(text="no disponible. Procese el archivo primero.",
                                        foreground=self._color_estado_calidad(None))
                labels["pct"].config(text="--")
                labels["pct_detalle"].config(text="Atípicos: -- | Sospechosos: --")
                labels["snr"].config(text="--")
                labels["region"].config(text="--")
                labels["atipicos"].config(text="--")
                labels["sospechosos"].config(text="--")
                continue

            metricas = self._metricas_calidad_resumen(resumen)
            estado = metricas["estado"]
            pct = metricas["pct_ponderado"]
            snr_db = metricas["snr_db"]
            region = metricas["region"]
            atipicos = self._formatear_lista_canales_compacta(resumen.get("canales_atipicos", []))
            sospechosos = self._formatear_lista_canales_compacta(resumen.get("canales_sospechosos", []))

            labels["estado"].config(text=f"Estado global: {estado}",
                                    foreground=self._color_estado_calidad(estado))
            labels["pct"].config(text=self._formatear_pct(pct))
            labels["pct_detalle"].config(
                text=(
                    f"Atípicos: {self._formatear_pct(metricas['pct_atipicos'])} | "
                    f"Sospechosos: {self._formatear_pct(metricas['pct_sospechosos'])}"
                )
            )
            labels["snr"].config(text=f"{snr_db:.2f} dB")
            labels["region"].config(text=region)
            labels["atipicos"].config(text=atipicos)
            labels["sospechosos"].config(text=sospechosos)

        self._actualizar_scrollregion_izquierda()

    def _actualizar_visibilidad_alertas_analisis(self):
        if not hasattr(self, "frm_alertas_analisis"):
            return

        if self.frm_alertas_analisis.winfo_ismapped():
            self.frm_alertas_analisis.pack_forget()

    def _actualizar_visibilidad_etiquetas_analisis(self):
        if not hasattr(self, "_frm_opciones_topo"):
            return

        contenedor = self._frm_opciones_topo
        tab = self._tab_analisis_actual()
        if tab == "calidad":
            if contenedor.winfo_ismapped():
                contenedor.pack_forget()
            return

        if not contenedor.winfo_ismapped():
            contenedor.pack(fill=tk.X, pady=(0, 8))

    def _mostrar_resumen_calidad_compacto_analisis(self):
        if not hasattr(self, "frm_resumen_calidad_analisis"):
            return

        if not self.frm_resumen_calidad_analisis.winfo_ismapped():
            antes = getattr(self, "_frm_opciones_topo", self.chk_etiquetas_topo)
            self.frm_resumen_calidad_analisis.pack(fill=tk.X, pady=(0, 8), before=antes)

    def _reset_resumen_visual_calidad(self, mensaje="Sin datos cargados."):
        if not hasattr(self, "lbl_calidad_dashboard_estado"):
            return

        self.lbl_calidad_dashboard_estado.config(
            text="Estado global: --",
            fg=self._color_estado_calidad(None)
        )
        self.lbl_calidad_dashboard_pct.config(text="--")
        self.lbl_calidad_dashboard_snr.config(text="--")
        self.lbl_calidad_dashboard_region.config(text="--")
        if hasattr(self, "lbl_calidad_dashboard_componentes"):
            self.lbl_calidad_dashboard_componentes.config(text="Atípicos: -- | Sospechosos: --")
        self.lbl_calidad_dashboard_snr_texto.config(text=mensaje)

    def _actualizar_resumen_visual_calidad(self, resumen):
        if not hasattr(self, "lbl_calidad_dashboard_estado"):
            return

        metricas = self._metricas_calidad_resumen(resumen)
        estado = metricas["estado"]
        pct = metricas["pct_ponderado"]
        snr_db = metricas["snr_db"]
        region = metricas["region"]
        interpretacion_snr = resumen.get("interpretacion_snr", "") or "Sin interpretación disponible."

        self.lbl_calidad_dashboard_estado.config(
            text=f"Estado global: {estado}",
            fg=self._color_estado_calidad(estado)
        )
        self.lbl_calidad_dashboard_pct.config(text=self._formatear_pct(pct))
        self.lbl_calidad_dashboard_snr.config(text=f"{snr_db:.2f} dB")
        self.lbl_calidad_dashboard_region.config(text=str(region))
        if hasattr(self, "lbl_calidad_dashboard_componentes"):
            self.lbl_calidad_dashboard_componentes.config(
                text=(
                    f"Atípicos: {self._formatear_pct(metricas['pct_atipicos'])} | "
                    f"Sospechosos: {self._formatear_pct(metricas['pct_sospechosos'])}"
                )
            )
        self.lbl_calidad_dashboard_snr_texto.config(text=f"{snr_db:.2f} dB | {interpretacion_snr}")
        if hasattr(self, "frm_calidad_resumen") and self.frm_calidad_resumen is not None:
            try:
                self.frm_calidad_resumen.update_idletasks()
            except Exception:
                pass
        self._actualizar_scrollregion_izquierda()

    def _puede_scroll_panel_izquierdo(self):
        if not hasattr(self, "canvas_izq") or self.canvas_izq is None:
            return False
        try:
            bbox = self.canvas_izq.bbox("all")
            if not bbox:
                return False
            alto_contenido = max(0, int(bbox[3] - bbox[1]))
            alto_canvas = max(1, int(self.canvas_izq.winfo_height()))
            return alto_contenido > (alto_canvas + 2)
        except Exception:
            return False

    def _actualizar_scrollregion_izquierda(self):
        if not hasattr(self, "canvas_izq") or self.canvas_izq is None or not hasattr(self, "_win_izq"):
            return

        try:
            self.canvas_izq.update_idletasks()
            self.canvas_izq.coords(self._win_izq, 0, 0)
            bbox = self.canvas_izq.bbox("all")
            if not bbox:
                return

            self.canvas_izq.configure(scrollregion=bbox)

            alto_canvas = max(1, int(self.canvas_izq.winfo_height()))
            alto_contenido = max(1, int(bbox[3] - bbox[1]))

            if alto_contenido <= (alto_canvas + 2):
                self.canvas_izq.yview_moveto(0.0)
                return

            first, _last = self.canvas_izq.yview()
            max_first = max(0.0, 1.0 - (alto_canvas / float(alto_contenido)))
            if first < 0.0:
                self.canvas_izq.yview_moveto(0.0)
            elif first > max_first:
                self.canvas_izq.yview_moveto(max_first)
        except Exception:
            pass

    def _hacer_colapsable(self, labelframe, pack_kwargs_body=None):
        """
        Mueve el título del LabelFrame a una fila header interna (misma línea
        que el botón toggle −/+). Retorna un Frame body con el contenido.
        """
        if pack_kwargs_body is None:
            pack_kwargs_body = {"fill": tk.X}

        # Leer y limpiar el título del borde del LabelFrame
        _PAD_EXP  = (6, 4, 6, 6)   # padding cuando está expandido
        _PAD_COL  = (6, 4, 6, 2)   # padding cuando está colapsado (menos abajo)
        try:
            titulo = labelframe.cget("text") or ""
            labelframe.configure(text="", padding=_PAD_EXP)
        except Exception:
            titulo = ""

        # Header: título a la izquierda, botón a la derecha — misma fila
        hdr = ttk.Frame(labelframe, style="Sidebar.TFrame")
        hdr.pack(fill=tk.X, pady=(0, 2))

        if titulo:
            ttk.Label(
                hdr, text=titulo,
                style="Sidebar.TLabel",
                font=(self.fuente_ui, 9, "bold"),
                foreground=AZUL_NEUROX,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        body = ttk.Frame(labelframe, style="Sidebar.TFrame")
        body.pack(**pack_kwargs_body)

        expandido = [True]

        def _toggle():
            if expandido[0]:
                body.pack_forget()
                btn.config(text="+")
                try:
                    labelframe.configure(padding=_PAD_COL)
                except Exception:
                    pass
            else:
                body.pack(**pack_kwargs_body)
                btn.config(text="−")
                try:
                    labelframe.configure(padding=_PAD_EXP)
                except Exception:
                    pass
            expandido[0] = not expandido[0]
            self.raiz.after_idle(self._actualizar_scrollregion_izquierda)

        btn = ttk.Button(hdr, text="−", width=2, command=_toggle, style="Secondary.TButton")
        btn.pack(side=tk.RIGHT)
        return body

    def _cargar_icono_btn(self, ruta_relativa, size=18):
        """Carga un PNG como ImageTk.PhotoImage para usar en botones."""
        try:
            ruta = resource_path(ruta_relativa)
            img = Image.open(ruta).convert("RGBA").resize((size, size), Image.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            return foto
        except Exception:
            return None

    def _formatear_ruta_panel(self, ruta, ancho=34):
        ruta = str(ruta or "").replace("\\", "/")
        if len(ruta) <= ancho:
            return ruta

        partes = textwrap.wrap(
            ruta,
            width=ancho,
            break_long_words=True,
            break_on_hyphens=False
        )
        return "\n".join(partes)

    def _iniciar_trabajo(self, txt):
        if self._cerrando:
            return
        self.set_progreso(0, txt)
        self.btn_sel.config(state="disabled")
        self.btn_proc.config(state="disabled")
        if hasattr(self, "_loading"):
            self._loading.show()

    def _terminar_trabajo(self, ocultar=False):
        if self._cerrando:
            return
        self.btn_sel.config(state="normal")
        self.btn_proc.config(state="normal")
        if hasattr(self, "_loading"):
            self._loading.hide()
        if ocultar:
            self.ocultar_progreso()

    ## =========================================================
    ## Limpieza de estado visual y datos activos
    ## =========================================================
    def _liberar_cache_visual(self):
        """
        Libera referencias de la GUI a arrays/gráficos del cache antes de Procesar.
        Ayuda a evitar bloqueos de archivos .npy en Windows.
        """
        try:
            # limpiar figura visible
            if hasattr(self, "fig") and self.fig is not None:
                self.fig.clear()
            if hasattr(self, "canvas") and self.canvas is not None:
                self.canvas.draw_idle()
        except Exception:
            pass
    
        # limpiar estado de vistas
        self._vista_actual = None
        self._ax_main = None
        self._line_canal = None
        self._lines_multicanal = []
        self._offsets_multicanal = None
    
        self._axes_bandas = []
        self._lines_bandas = []
        self._bandas_actuales = []
        self._idx_actual = None
        self._canal_actual = None
        self._nombre_actual = None
    
        # análisis
        self._er_filtrado = None
        self._pos_1010 = {}
        self._limpiar_estado_welch()
    
        # sliders / ejes auxiliares
        try:
            self._limpiar_controles_tiempo()
        except Exception:
            pass
    
        import gc
        gc.collect()

    def _limpiar_estado_welch(self):
        self._pot_rel_welch = None
        self._resumen_welch = None
        self._welch_topo_xy = {}
        self._ax_welch_to_band = {}
        self.cbar_welch = {}
        self.canal_potencia_welch_simple = None
        self._limpiar_estado_ratios()
        self._limpiar_estado_calidad()

        self._reset_panel_izquierdo_potencia_relativa()

        if hasattr(self, "fig_welch") and self.fig_welch is not None:
            try:
                self.fig_welch.clear()
                if hasattr(self, "canvas_welch") and self.canvas_welch is not None:
                    self.canvas_welch.draw_idle()
            except Exception:
                pass

    def _limpiar_estado_ratios(self):
        self._ratios_principales = None
        self._resumen_ratios = None
        self._ratio_personalizado = None
        self._ratios_topo_xy = {}
        self._ax_ratios_to_nombre = {}
        self.cbar_ratios = {}
        self.canal_ratios_simple = None
        self._ratio_personalizado_activo = False

        if hasattr(self, "var_ratio_num"):
            try:
                self.var_ratio_num.set("Theta")
                self.var_ratio_den.set("Alfa")
            except Exception:
                pass

        self._reset_panel_izquierdo_ratios()

        if hasattr(self, "fig_ratios") and self.fig_ratios is not None:
            try:
                self.fig_ratios.clear()
                if hasattr(self, "canvas_ratios") and self.canvas_ratios is not None:
                    self.canvas_ratios.draw_idle()
            except Exception:
                pass

    def _limpiar_estado_calidad(self):
        self._resumen_calidad = None
        self._texto_calidad = None
        self._reset_panel_izquierdo_calidad()

        if hasattr(self, "fig_calidad") and self.fig_calidad is not None:
            try:
                self.fig_calidad.clear()
                if hasattr(self, "canvas_calidad") and self.canvas_calidad is not None:
                    self.canvas_calidad.draw_idle()
            except Exception:
                pass
        
    def limpiar_datos(self):
        """
        Limpia el archivo/caché activo y reinicia la visualización,
        sin cerrar la app ni perder la carpeta cargada.
        """
        # detener autoactualización pendiente
        if self._after_autograf is not None:
            try:
                self.raiz.after_cancel(self._after_autograf)
            except Exception:
                pass
            self._after_autograf = None
    
        # liberar referencias visuales
        self._liberar_cache_visual()
    
        # reset de archivo/cache activo
        self.cache_ultimo = None
        self.nombre_ultimo = None
        self.ruta_pdf_ultimo = None
    
        # reset de estado del visor/análisis
        self.canal_activo_simple = None
        self.canal_analisis_simple = None
        self.var_atipicos.set("Canales atípicos: --")
        self.var_sospechosos.set("Canales sospechosos: --")
        self.hay_grafico_activo = False
    
        self._fs = None
        self._tf_total = None
        self._t0 = 0.0
        self._dt = 5.0
    
        self._er_filtrado = None
        self._pos_1010 = {}
        self._limpiar_estado_welch()
        if hasattr(self, "_pos_1010_multi"):
            self._pos_1010_multi = {}
    
        # limpiar selección de archivo
        try:
            for _sel in self.lista.selection():
                self.lista.selection_remove(_sel)
        except Exception:
            pass
    
        # reset de controles
        try:
            self.combo_canal.current(0)
            self.var_canal_nombre.set(self.nombres_canales[0])
        except Exception:
            pass
    
        self.var_modo.set("Filtrado multicanal")
        self.var_ganancia.set(1.0)
        self.var_ventana.set(5.0)
        self.var_fmax.set(40.0)
        self.var_ventana_espectrograma.set(2.0)
        self.var_solapamiento_espectrograma.set(50.0)
    
        for b in self.bandas:
            self.band_vars[b].set(b == "Alfa")
    
        self._refrescar_ui_modo()
        self._mostrar_marco_visor_grafica(False)
        self._mostrar_tabs_grafica(False)
    
        # desactivar botones dependientes del archivo procesado
        self.btn_pdf.config(state="disabled")
        self.btn_proc.config(state="disabled")
        self.btn_tab_analisis.config(style="Secondary.TButton")
        self.btn_tab_grafica.config(style="Accent.TButton")
    
        # limpiar panel derecho (visor)
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "Sin datos cargados",
                ha="center", va="center",
                fontsize=12
            )
            self.canvas.draw_idle()
        except Exception:
            pass
    
       
        # reset panel de análisis
        self._reset_panel_analisis()
        self._actualizar_bloques_resumen_calidad()
        # try:
        #     self._dibujar_topomap_analisis()
        # except Exception:
        #     pass
    
        # volver a visor
        self.ir_a_visor()
    
        # mensajes
        self._set_barra("Datos limpiados. Selecciona un archivo para continuar.")
        self.log("Datos limpiados.")
        
    def _reset_panel_analisis(self):
        try:
            self.lbl_sel_canal.config(text="Canal: (haz clic en el mapa)")
            self.lbl_region.config(text="Región: --")
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                self.lbl_bandas[b].config(text=f"{b}: --%")
                self.lbl_region_bandas[b].config(text=f"{b} (región): --%")
            self._reset_panel_izquierdo_potencia_relativa()
            self._reset_panel_izquierdo_ratios()
            self._reset_panel_izquierdo_calidad()
            self._mostrar_panel_izq_analisis_actual()
        except Exception:
            pass

    ## =========================================================
    ## Canales, alertas y nombres EEG
    ## =========================================================
    def _idxs_a_nombres(self, arr_idx):
        arr_idx = np.asarray(arr_idx).astype(int).ravel()
        if arr_idx.size == 0:
            return []
    
        nombres = []
    
        # si parecen venir base 0
        if np.max(arr_idx) < len(self.nombres_canales):
            for i in arr_idx:
                if 0 <= i < len(self.nombres_canales):
                    nombres.append(self._nombre_simple(self.nombres_canales[i]))
        else:
            # si vinieran base 1
            for i in arr_idx:
                j = i - 1
                if 0 <= j < len(self.nombres_canales):
                    nombres.append(self._nombre_simple(self.nombres_canales[j]))
    
        return nombres
    
    
    def _actualizar_canales_alerta(self):
        self.var_atipicos.set("Canales atípicos: --")
        self.var_sospechosos.set("Canales sospechosos: --")
    
        if not self.cache_ultimo:
            self._actualizar_bloques_resumen_calidad()
            return
    
        carpeta = os.path.join(self.cache_ultimo, "canales_atipicos")
        if not os.path.isdir(carpeta):
            self.log(f"No existe carpeta de alertas: {carpeta}")
            self._actualizar_bloques_resumen_calidad()
            return
    
        def cargar_nombres(candidatos_exactos, palabra_clave):
            ruta = None
    
            for nombre in candidatos_exactos:
                r = os.path.join(carpeta, nombre)
                if os.path.exists(r):
                    ruta = r
                    break
    
            if ruta is None:
                for nom in os.listdir(carpeta):
                    nl = nom.lower()
                    if palabra_clave in nl and nl.endswith(".npy"):
                        ruta = os.path.join(carpeta, nom)
                        break
    
            if ruta is None:
                return None
    
            try:
                arr = np.load(ruta, allow_pickle=True)
                return self._idxs_a_nombres(arr)
            except Exception as e:
                self.log(f"No pude leer {ruta}: {e}")
                return None
    
        nombres_atipicos = cargar_nombres(
            [
                "canales_atipicos_fuertes_idx0.npy",
                "canales_atipicos_fuertes_idx1.npy"
            ],
            "atip"
        )
    
        nombres_sospechosos = cargar_nombres(
            [
                "canales_sospechosos_idx0.npy",
                "canales_sospechosos_idx1.npy"
            ],
            "sospech"
        )
    
        if nombres_atipicos is None:
            self.var_atipicos.set("Canales atípicos: --")
        elif nombres_atipicos:
            self.var_atipicos.set("Canales atípicos: " + ", ".join(nombres_atipicos))
        else:
            self.var_atipicos.set("Canales atípicos: ninguno")
    
        if nombres_sospechosos is None:
            self.var_sospechosos.set("Canales sospechosos: --")
        elif nombres_sospechosos:
            self.var_sospechosos.set("Canales sospechosos: " + ", ".join(nombres_sospechosos))
        else:
            self.var_sospechosos.set("Canales sospechosos: ninguno")

        self._actualizar_bloques_resumen_calidad()
            
        
        
    def _cargar_canales_64(self):
        try:
            return obtener_nombres_canales_64()
        except Exception:
            return [f"Canal {i+1}" for i in range(64)]

    ## =========================================================
    ## Controles del visor y actualización automática
    ## =========================================================
    def _mostrar_marco_visor_grafica(self, visible):
        if not hasattr(self, "marco_visor"):
            return

        visible = bool(visible)
        if visible == getattr(self, "_marco_visor_visible", False):
            return

        if visible:
            after_widget = getattr(self, "_calidad_wrapper_grafica",
                           getattr(self, "frm_resumen_calidad_grafica", self.lbl_sospechosos_grafica))
            self.marco_visor.pack(fill=tk.X, after=after_widget)
        else:
            self.marco_visor.pack_forget()

        self._marco_visor_visible = visible

    def _normalizar_modo_grafica(self, modo):
        equivalencias = {
            "Multicanal": "Filtrado multicanal",
            "multicanal": "Filtrado multicanal",
            "Canal Filtrado": "Canal filtrado",
            "FFT": "FFT filtrada",
            "Espectrograma": "Espectrograma",
            "espectrograma": "Espectrograma",
        }
        return equivalencias.get(modo, modo)

    def _mostrar_tabs_grafica(self, visible):
        target = getattr(self, "_frame_tabs_grafica", None) or getattr(self, "tabs_grafica", None)
        if target is None:
            return

        visible = bool(visible)
        if visible == getattr(self, "_tabs_grafica_visible", False):
            return

        if visible:
            target.pack(fill=tk.X, before=self.frame_visor_plot)
        else:
            target.pack_forget()

        self._tabs_grafica_visible = visible

    def _tab_desde_modo(self, modo):
        modo = self._normalizar_modo_grafica(modo)
        for texto_tab, modo_interno in self.modos_grafica_tabs.items():
            if modo_interno == modo:
                return texto_tab
        return "Multicanal"

    def _modo_desde_tab_actual(self):
        if not hasattr(self, "tabs_grafica"):
            return self._normalizar_modo_grafica(self.var_modo.get())

        tab_id = self.tabs_grafica.select()
        if not tab_id:
            return self._normalizar_modo_grafica(self.var_modo.get())

        texto_tab = self.tabs_grafica.tab(tab_id, "text")
        return self.modos_grafica_tabs.get(texto_tab, self._normalizar_modo_grafica(self.var_modo.get()))

    def _sincronizar_tab_grafica(self):
        if not hasattr(self, "tabs_grafica"):
            return

        texto_tab = self._tab_desde_modo(self.var_modo.get())
        frame_tab = self.frames_tabs_grafica.get(texto_tab)
        if frame_tab is None:
            return

        if self.tabs_grafica.select() == str(frame_tab):
            return

        self._sincronizando_tab_grafica = True
        try:
            self.tabs_grafica.select(frame_tab)
        finally:
            self._sincronizando_tab_grafica = False
        self.actualizar_nav_canal()

    def _on_tab_grafica(self, _evt=None):
        if getattr(self, "_sincronizando_tab_grafica", False):
            return

        modo = self._modo_desde_tab_actual()
        if modo != self.var_modo.get():
            self.var_modo.set(modo)

        self._refrescar_ui_modo()
        self.actualizar_nav_canal()

        if not self.cache_ultimo:
            self._mostrar_placeholder_visor()
            return

        self.hay_grafico_activo = True
        if modo not in ("FFT filtrada", "Espectrograma"):
            if self._cid_press is None or self._cid_motion is None or self._cid_release is None:
                self._activar_pan_tiempo()
        self._redibujar_modo_actual()

    def _refrescar_ui_modo(self, _evt=None):
        modo = self._normalizar_modo_grafica(self.var_modo.get())
        if modo != self.var_modo.get():
            self.var_modo.set(modo)
        self._sincronizar_tab_grafica()
    
        self.frame_canal.pack_forget()
        self.frame_ganancia.pack_forget()
        self.frame_ventana.pack_forget()
        self.frame_fft.pack_forget()
        self.frame_espectrograma.pack_forget()
        self.frame_bandas.pack_forget()
        self.frame_mapa.pack_forget()

        self.lbl_fmax.config(text="FFT hasta (Hz):")
        self.spin_fmax.configure(from_=20.0, to=100.0, increment=5.0)
        self.btn_graficar.config(text="Graficar")
    
        if modo == "Filtrado multicanal":
            self.frame_ganancia.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
            self.frame_ventana.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
    
        elif modo == "FFT filtrada":
            self.frame_canal.pack(fill=tk.X, before=self.btn_graficar)
            self.frame_fft.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
            self.frame_mapa.pack(fill=tk.X, pady=(8, 0))
            self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
    
        elif modo == "Canal filtrado":
            self.frame_canal.pack(fill=tk.X, before=self.btn_graficar)
            self.frame_ventana.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
            self.frame_mapa.pack(fill=tk.X, pady=(8, 0))
            self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
    
        elif modo == "Bandas":
            self.frame_canal.pack(fill=tk.X, before=self.btn_graficar)
            self.frame_ventana.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
            self.frame_bandas.pack(fill=tk.X, pady=(0, 0))
            self.frame_mapa.pack(fill=tk.X, pady=(8, 0))
            self.raiz.after_idle(self._dibujar_mapa_regional_resumen)

        elif modo == "Espectrograma":
            self.frame_canal.pack(fill=tk.X, before=self.btn_graficar)
            self.frame_fft.pack(fill=tk.X, pady=(0, 0), before=self.btn_graficar)
            self.frame_mapa.pack(fill=tk.X, pady=(8, 0))
            self.lbl_fmax.config(text="Frecuencia máxima (Hz):")
            self.spin_fmax.configure(from_=1.0, to=40.0, increment=1.0)
            try:
                if float(self.var_fmax.get()) > 40.0:
                    self.var_fmax.set(40.0)
            except Exception:
                self.var_fmax.set(40.0)
            self.btn_graficar.config(text="Graficar espectrograma")
            self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
        self.actualizar_nav_canal()
            
    def _programar_autografico(self, *_args):
        """
        Programa una actualización automática 1 segundo después
        del último cambio en controles como ganancia, ventana o FFT.
        """
        if self._cerrando:
            return
    
        # cancelar temporizador anterior
        if self._after_autograf is not None:
            try:
                self.raiz.after_cancel(self._after_autograf)
            except Exception:
                pass
            self._after_autograf = None
    
        # nuevo temporizador
        self._after_autograf = self.raiz.after(1000, self._autografico_ejecutar)
    
    
    def _autografico_ejecutar(self):
        self._after_autograf = None
    
        if self._cerrando or not self.cache_ultimo:
            return
    
        try:
            modo = self._normalizar_modo_grafica(self.var_modo.get())
            if modo != self.var_modo.get():
                self.var_modo.set(modo)
    
            if not self.hay_grafico_activo:
                return
    
            # usar redibujado liviano cuando sea posible
            if modo == "Filtrado multicanal":
                if self._vista_actual == "multicanal" and self._lines_multicanal:
                    self._actualizar_filtrado_multicanal()
                else:
                    self._graficar_filtrado_multicanal()
    
            elif modo == "Canal filtrado":
                if self._vista_actual == "canal" and self._line_canal is not None:
                    self._actualizar_canal_filtrado()
                else:
                    idx = self.combo_canal.current()
                    if idx < 0:
                        idx = 0
                    canal = idx + 1
                    nombre = self.nombres_canales[idx]
                    self._graficar_canal_filtrado(idx, canal, nombre)
    
            elif modo == "FFT filtrada":
                idx = self.combo_canal.current()
                if idx < 0:
                    idx = 0
                canal = idx + 1
                nombre = self.nombres_canales[idx]
                self._graficar_fft_filtrada(idx, canal, nombre)
    
            elif modo == "Bandas":
                if self._vista_actual == "bandas" and self._lines_bandas:
                    self._actualizar_bandas_multi()
                else:
                    idx = self.combo_canal.current()
                    if idx < 0:
                        idx = 0
                    canal = idx + 1
                    nombre = self.nombres_canales[idx]
                    self._graficar_bandas_multi(idx, canal, nombre)

            elif modo == "Espectrograma":
                idx = self.combo_canal.current()
                if idx < 0:
                    idx = 0
                canal = idx + 1
                nombre = self.nombres_canales[idx]
                self._graficar_espectrograma(idx, canal, nombre)
    
        except Exception as e:
            self.log(f"Autoactualización: {e}")        

    ## =========================================================
    ## Mapa regional 10-10/10-20
    ## =========================================================
    def _seleccionar_canal_por_nombre(self, nombre_simple: str):
        """
        nombre_simple: ejemplo 'FP1' o 'OZ' o 'M1'
        Busca en self.nombres_canales algo tipo '' y lo selecciona.
        """
        objetivo = self._nombre_simple(nombre_simple)
    
        # busca índice
        idx = None
        for i, n in enumerate(self.nombres_canales):
            if self._nombre_simple(n) == objetivo:
                idx = i
                break
    
        if idx is None:
            messagebox.showwarning("Mapa 10–20", f"No encontré {objetivo} en la lista de 64 canales.")
            return
    
        self.combo_canal.current(idx)
        self.var_canal_nombre.set(self.nombres_canales[idx])
        self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
        self.actualizar_nav_canal()
        self.graficar()
        
    def _color_region(self, etiqueta):
        e = etiqueta.upper()
    
        # Frontal
        if e.startswith(("FP", "AF", "F", "FT", "FC")):
            return "#9AD0F5", "Frontal"
    
        # Central / Temporal
        if e.startswith(("C", "T")):
            return "#A7F3D0", "Central/Temporal"
    
        # Parietal
        if e.startswith(("TP", "CP", "P")):
            return "#C4B5FD", "Parietal"
    
        # Occipital / Cerebelo
        if e.startswith(("PO", "O", "CB")):
            return "#FCA5A5", "Occipital/Cerebelo"
    
        # Mastoides (orejas)
        if e in ("M1", "M2", "A1", "A2"):
            return "#D1D5DB", "Mastoides"
    
        return "#E5E7EB", "Otro"
    
    def _dibujar_punto(self, x, y, etiqueta, r=10):
        tag = f"EL_{etiqueta}"
    
        color, _ = self._color_region(etiqueta)
    
        self.canvas_mapa.create_oval(
            x-r, y-r, x+r, y+r,
            fill=color, outline="#2b2b2b", width=1,
            tags=(tag,)
        )
        self.canvas_mapa.create_text(
            x, y, text=etiqueta, font=("Segoe UI", 7, "bold"),
            tags=(tag,)
        )

        self.canvas_mapa.tag_bind(tag, "<Button-1>", lambda e, lab=etiqueta: self._seleccionar_canal_por_nombre(lab))

    def _codigo_region(self, etiqueta):
        e = etiqueta.upper()
    
        if e.startswith(("FP", "AF", "F", "FT", "FC")):
            return 1   # frontal
        if e.startswith(("C", "T")):
            return 2   # central/temporal
        if e.startswith(("TP", "CP", "P")):
            return 3   # parietal
        if e.startswith(("PO", "O", "CB")):
            return 4   # occipital/cerebelo
        if e in ("M1", "M2", "A1", "A2"):
            return 5   # mastoides
    
        return 0

    def _dibujar_mapa_regional_resumen(self):
        if self._cerrando:
            return
    
        if not hasattr(self, "canvas_mapa") or self.canvas_mapa is None:
            return
    
        c = self.canvas_mapa
    
        if not c.winfo_exists():
            return
    
        w = c.winfo_width()
        h = c.winfo_height()
    
        if w < 20 or h < 20:
            return
        c.delete("all")
    
        w = max(1, c.winfo_width())
        h = max(1, c.winfo_height())
    
        cx, cy = w // 2, h // 2
        R = int(min(w, h) * 0.40)
        R = max(85, R)
    
        rr_el = max(4, int(R * 0.038))
        font_sz = max(5, int(R * 0.052))
        font_el = ("Segoe UI", font_sz, "bold")
        canal_seleccionado = self._nombre_simple(self.var_canal_nombre.get()) if hasattr(self, "var_canal_nombre") else ""
    
        disponibles = sorted(set([n.upper().replace("-AVG", "") for n in self.nombres_canales]))
    
        # =====================================================
        # FONDO REGIONAL POR MÁSCARAS GEOMÉTRICAS
        # =====================================================
        # imagen RGBA
        ngrid = 320
        gx = np.linspace(cx - R, cx + R, ngrid)
        gy = np.linspace(cy - R, cy + R, ngrid)
        XI, YI = np.meshgrid(gx, gy)
    
        # coordenadas normalizadas respecto a la cabeza
        XN = (XI - cx) / R
        YN = (YI - cy) / R
    
        # máscara exacta del círculo
        mask_head = (XN**2 + YN**2) <= 0.985**2
    
        # imagen base blanca
        img = np.ones((ngrid, ngrid, 4), dtype=np.uint8) * 255
    
        # colores
        col_frontal   = np.array([154, 208, 245, 255], dtype=np.uint8)
        col_central   = np.array([167, 243, 208, 255], dtype=np.uint8)
        col_parietal  = np.array([196, 181, 253, 255], dtype=np.uint8)
        col_occipital = np.array([252, 165, 165, 255], dtype=np.uint8)
    
        # -------------------------
        # REGIONES
        # -------------------------
        # Frontal
        m_frontal = (
            mask_head &
            (YN <= -0.20)
        )
        
        # Central/Temporal
        m_central = (
            mask_head &
            (YN > -0.20) & (YN <= 0.28)
        )
        
        m_temporal_izq = (
            mask_head &
            (((XN + 0.88) / 0.28) ** 2 + ((YN - 0.02) / 0.42) ** 2 <= 1.0)
        )
        m_temporal_der = (
            mask_head &
            (((XN - 0.88) / 0.28) ** 2 + ((YN - 0.02) / 0.42) ** 2 <= 1.0)
        )
        
        # Parietal
        m_parietal = (
            mask_head &
            (YN > 0.28) & (YN <= 0.78)
        )
        
        # Occipital/Cerebelo
        # m_occipital = (
        #     mask_head &
        #     (((XN / 0.52) ** 2 + ((YN - 0.84) / 0.24) ** 2) <= 1.0)
        # )
        
        m_occipital = (
            mask_head &
            (((XN / 0.62) ** 2 + ((YN - 0.80) / 0.30) ** 2) <= 1.0)
        )
        # -------------------------
        # Pintar en orden
        # -------------------------
        img[m_frontal] = col_frontal
        img[m_central] = col_central
        img[m_temporal_izq] = col_central
        img[m_temporal_der] = col_central
        img[m_parietal] = col_parietal
        img[m_occipital] = col_occipital
    
        # fuera de la cabeza = blanco
        img[~mask_head] = (255, 255, 255, 255)
    
        lado = int(2 * R)

        pil_img = Image.fromarray(img, mode="RGBA")
        pil_img = pil_img.resize((lado, lado), Image.Resampling.LANCZOS)
        
        self._img_mapa_regional_tk = ImageTk.PhotoImage(pil_img, master=self.canvas_mapa)
        
        c.create_image(
            int(cx - R), int(cy - R),
            image=self._img_mapa_regional_tk,
            anchor="nw"
        )
            
        # =====================================================
        # MASTOIDES
        # =====================================================
        for et in ("M1", "M2"):
            p = self._coord_1010_curvada(et, cx, cy, R)
            if p is None:
                continue
            x, y = p
            color, _ = self._color_region(et)
            rr = max(8, int(R * 0.085))
            c.create_oval(x - rr, y - rr, x + rr, y + rr, fill=color, outline="")
    
        # =====================================================
        # CONTORNOS DE CABEZA
        # =====================================================
        c.create_oval(cx - R, cy - R, cx + R, cy + R, outline="#555", width=2)
    
        c.create_polygon(
            cx - 10, cy - R + 8,
            cx + 10, cy - R + 8,
            cx,      cy - R - 14,
            outline="#555", fill="", width=2
        )
    
        ear_w = int(0.12 * R)
        ear_h = int(0.18 * R)
        c.create_oval(cx - R - ear_w // 2, cy - ear_h // 2,
                      cx - R + ear_w // 2, cy + ear_h // 2,
                      outline="#777", width=2)
        c.create_oval(cx + R - ear_w // 2, cy - ear_h // 2,
                      cx + R + ear_w // 2, cy + ear_h // 2,
                      outline="#777", width=2)
    
        # guías curvas
        # for s, col in [
        #     (0.10, "#D5D5D5"), (0.20, "#D5D5D5"), (0.30, "#C8C8C8"),
        #     (0.40, "#D0D0D0"), (0.50, "#B8B8B8"),
        #     (0.60, "#D0D0D0"), (0.70, "#C8C8C8"), (0.80, "#D5D5D5"),
        #     (0.90, "#D5D5D5"),
        # ]:
        #     self._dibujar_guia_fila_curvada(c, cx, cy, R, s, color=col)
    
        # =====================================================
        # ELECTRODOS NEGROS + TEXTO
        # =====================================================
        for et in disponibles:
            p = self._coord_1010_curvada(et, cx, cy, R)
            if p is None:
                continue
    
            x, y = p
            tag = f"MAPA_REG_{et}"
            seleccionado = (self._nombre_simple(et) == canal_seleccionado)

            if seleccionado:
                c.create_oval(
                    x - rr_el - 4, y - rr_el - 4, x + rr_el + 4, y + rr_el + 4,
                    fill="#2563EB",
                    outline="white",
                    width=2,
                    tags=(tag,)
                )
    
            c.create_oval(
                x - rr_el, y - rr_el, x + rr_el, y + rr_el,
                fill="white" if seleccionado else "black",
                outline="#2563EB" if seleccionado else "black",
                width=1,
                tags=(tag,)
            )
    
            c.create_text(
                x, y - rr_el - 5,
                text=et,
                font=font_el,
                fill="#1D4ED8" if seleccionado else "black",
                tags=(tag,)
            )
    
            c.tag_bind(tag, "<Button-1>", lambda e, lab=et: self._seleccionar_canal_por_nombre(lab))
            
            
                    
    def abrir_mapa_grande(self):
        win = tk.Toplevel(self.raiz)
        win.title("Mapa 10–20 (64 canales)")
        win.geometry("860x820")
    
        frm = ttk.Frame(win, padding=10, style="NeuroX.TFrame")
        frm.pack(fill=tk.BOTH, expand=True)
    
        barra = ttk.Frame(frm, style="NeuroX.TFrame")
    
        canvas = tk.Canvas(frm, bg=FONDO_PANEL, highlightthickness=1, highlightbackground=BORDE_SUAVE)
        sx = ttk.Scrollbar(frm, orient="horizontal", command=canvas.xview, style="NeuroX.Horizontal.TScrollbar")
        sy = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview, style="NeuroX.Vertical.TScrollbar")
        canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
    
        # GRID estable
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)
        barra.grid(row=0, column=0, columnspan=2, sticky="ew")
        canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        sy.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        sx.grid(row=2, column=0, sticky="ew")
    
        estado = {"scale": 1.0, "centrado_inicial": False}
    
        def clamp(v, a, b):
            return a if v < a else b if v > b else v
        
        def centrar_por_bbox():
            bb = canvas.bbox("all")
            if not bb:
                return
            x0, y0, x1, y1 = bb
            canvas.configure(scrollregion=bb)
        
            vw = max(1, canvas.winfo_width())
            vh = max(1, canvas.winfo_height())
        
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
        
            # centro del bbox
            cx = x0 + bw / 2
            cy = y0 + bh / 2
        
            # queremos que ese centro quede en el centro de la pantalla
            left = cx - vw / 2
            top  = cy - vh / 2
        
            fx = 0.0 if bw <= vw else left / (bw - vw)
            fy = 0.0 if bh <= vh else top  / (bh - vh)
        
            canvas.xview_moveto(clamp(fx, 0.0, 1.0))
            canvas.yview_moveto(clamp(fy, 0.0, 1.0))
        def centrar_vista():
            # centra según scrollregion REAL y tamaño visible REAL
            sr = canvas.cget("scrollregion")
            if not sr:
                return
            x0, y0, x1, y1 = map(float, sr.split())
            vw = max(1, canvas.winfo_width())
            vh = max(1, canvas.winfo_height())
    
            bw = max(1.0, x1 - x0)
            bh = max(1.0, y1 - y0)
    
            left = (bw - vw) / 2
            top  = (bh - vh) / 2
    
            fx = 0.0 if bw <= vw else left / (bw - vw)
            fy = 0.0 if bh <= vh else top  / (bh - vh)
    
            canvas.xview_moveto(clamp(fx, 0.0, 1.0))
            canvas.yview_moveto(clamp(fy, 0.0, 1.0))
    
        def dibujar_full(centrar=True):
            canvas.delete("all")
            self._dibujar_mapa_full_en_canvas(canvas, scale=estado["scale"])
        
            canvas.update_idletasks()
            win.update_idletasks()
        
            if centrar:
                centrar_por_bbox()
    
        def zoom_in():
            estado["scale"] *= 1.15
            dibujar_full(centrar=True)
        
        def zoom_out():
            estado["scale"] /= 1.25
            dibujar_full(centrar=True)
        
        def reset():
            estado["scale"] = 1.0
            dibujar_full(centrar=True)
        def ajustar_a_ventana():
            win.update_idletasks()
            canvas.update_idletasks()
            vw = max(1, canvas.winfo_width())
            vh = max(1, canvas.winfo_height())
            base_w, base_h = 1400, 1100
            estado["scale"] = 0.95 * min(vw / base_w, vh / base_h)
            dibujar_full(centrar=True)

        ttk.Button(barra, text="Ajustar", command=ajustar_a_ventana).pack(side=tk.LEFT, padx=4)
        ttk.Button(barra, text="Zoom +", command=zoom_in).pack(side=tk.LEFT, padx=4)
        ttk.Button(barra, text="Zoom -", command=zoom_out).pack(side=tk.LEFT, padx=4)
        ttk.Button(barra, text="Reset", command=reset).pack(side=tk.LEFT, padx=4)
    
        # -------------------------
        # Pan con botón DERECHO (no interfiere con seleccionar canal)
        # -------------------------
        def start_pan(event):
            canvas.scan_mark(event.x, event.y)
    
        def do_pan(event):
            canvas.scan_dragto(event.x, event.y, gain=1)
    
        canvas.bind("<ButtonPress-3>", start_pan)
        canvas.bind("<B3-Motion>", do_pan)
    
        # -------------------------
        # Zoom con rueda del mouse (manteniendo el punto bajo el cursor)
        # -------------------------
        def zoom_rueda(factor, event):
            # medidas antes del zoom
            sr = canvas.cget("scrollregion")
            if not sr:
                return
            x0, y0, x1, y1 = map(float, sr.split())
            old_w = max(1.0, x1 - x0)
            old_h = max(1.0, y1 - y0)
    
            vw = max(1, canvas.winfo_width())
            vh = max(1, canvas.winfo_height())
    
            # coordenada del cursor en el "mundo" del canvas
            wx = canvas.canvasx(event.x)
            wy = canvas.canvasy(event.y)
    
            ax = wx / old_w
            ay = wy / old_h
    
            # aplicar zoom
            estado["scale"] *= factor
            dibujar_full(centrar=False)
    
            # medidas después del zoom
            sr2 = canvas.cget("scrollregion")
            x0b, y0b, x1b, y1b = map(float, sr2.split())
            new_w = max(1.0, x1b - x0b)
            new_h = max(1.0, y1b - y0b)
    
            # queremos que (ax, ay) quede bajo el cursor (event.x, event.y)
            target_x = ax * new_w - event.x
            target_y = ay * new_h - event.y
    
            fx = 0.0 if new_w <= vw else target_x / (new_w - vw)
            fy = 0.0 if new_h <= vh else target_y / (new_h - vh)
    
            canvas.xview_moveto(clamp(fx, 0.0, 1.0))
            canvas.yview_moveto(clamp(fy, 0.0, 1.0))
    
        def on_mousewheel(event):
            # Windows: event.delta (±120)
            if event.delta > 0:
                zoom_rueda(1.12, event)
            else:
                zoom_rueda(1/1.12, event)
    
        # Windows / Mac
        canvas.bind("<MouseWheel>", on_mousewheel)
    
        # Linux (algunas distros usan Button-4/5)
        canvas.bind("<Button-4>", lambda e: zoom_rueda(1.12, e))
        canvas.bind("<Button-5>", lambda e: zoom_rueda(1/1.12, e))
    
        # -------------------------
        # Centrando AL FINAL cuando ya todo está dibujado y dimensionado
        # -------------------------
        def centrar_al_abrir():
            dibujar_full(centrar=True)
            estado["centrado_inicial"] = True
    
        # Espera un poquito para que el canvas tenga tamaño real
        win.after(350, centrar_al_abrir)

        # Sync con canal activo: redibuja el mapa grande cuando cambia el canal
        def _sync_canal_mapa_grande(*_):
            if win.winfo_exists():
                dibujar_full(centrar=False)
        trace_id = self.var_canal_nombre.trace_add("write", _sync_canal_mapa_grande)

        def _al_cerrar():
            try:
                self.var_canal_nombre.trace_remove("write", trace_id)
            except Exception:
                pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _al_cerrar)

        # Si cambias el tamaño de la ventana, recentra SOLO la primera vez o en reset
        def _on_resize(_evt=None):
            if not estado["centrado_inicial"]:
                return

        canvas.bind("<Configure>", _on_resize)
        
    
    

    def _s_por_prefijo_1010(self, pref: str) -> float:
        """Porcentaje (0..1) a lo largo de Nz->Iz para cada 'fila' del 10-10."""
        pref = pref.upper()
        if pref == "FP":  return 0.10
        if pref == "AF":  return 0.20
        if pref == "F":   return 0.31
        if pref in ("FC", "FT"): return 0.40
        if pref in ("C", "T"):   return 0.50
        if pref in ("CP", "TP"): return 0.60
        if pref == "P":   return 0.70
        if pref == "PO":  return 0.80
        if pref == "O":   return 0.90
        return 0.50
    
    def _lateral_por_numero(self, pref: str, n: int) -> float:
        """Qué tan lateral (0..1) es un electrodo según su número."""
        pref = pref.upper()
        # escala base
        if n in (1, 2):   lf = 0.22
        elif n in (3, 4): lf = 0.42
        elif n in (5, 6): lf = 0.62
        elif n in (7, 8): lf = 0.92
        elif n in (9, 10): lf = 0.94
        else: lf = 0.50
    
        # temporales/laterales suelen ir aún más hacia la oreja
        if pref in ("T", "FT", "TP") and n in (7, 8):
            lf = 0.95
        # --- ajuste: separar más los pares 1/2 en FP y O (FP1/FP2, O1/O2)
        if pref in ("FP", "O") and n in (1, 2):
            lf = 0.44   # antes ~0.22 -> prueba 0.28 a 0.34
        return lf
    
    def _t_uniforme_en_fila(self, pref: str, suf: str):
        """
        Devuelve t en [-1..1] según el ORDEN REAL de la fila (espaciado uniforme).
        Si no aplica, retorna None.
        """
        pref = pref.upper()
        suf = suf.upper()
    
        orden_por_fila = {
            # Fila F: F7 F5 F3 F1 Fz F2 F4 F6 F8 (equidistantes)
            "F":  ["7","5","3","1","Z","2","4","6","8"],
            "FC": ["5", "3", "1", "Z", "2", "4", "6" ],
            "C":  ["5","3","1","Z","2","4","6"],
            "CP":  ["5","3","1","Z","2","4","6"],
            
    
            # Fila P: P7 P5 P3 P1 Pz P2 P4 P6 P8 (equidistantes)
            "P":  ["7","5","3","1","Z","2","4","6","8"],
    
            # Fila PO: PO7 PO5 PO3 POz PO4 PO6 PO8 (equidistantes)
            "PO": ["7","5","3","Z","4","6","8"],
            
        }
    
        if pref not in orden_por_fila:
            return None
    
        orden = orden_por_fila[pref]
        if suf not in orden:
            return None
    
        i = orden.index(suf)
        mid = (len(orden)-1)/2.0
        # normaliza a [-1..1]
        t = (i - mid) / mid
        return t
    
    def _factor_apertura_fila(self, pref: str) -> float:
        """
        >1.0 = abre (separa más)
        <1.0 = cierra (mete laterales hacia adentro)
        """
        pref = pref.upper()
    
        # Queremos separar FP1–FPZ–FP2 y O1–OZ–O2
        if pref in ("FP", "O"):
            return 1.15
    
        # Estas filas chocan con temporales (T/FT/TP), así que las cerramos
        if pref in ("C", "FC", "CP"):
            return 0.82
    
        # F y P un poquito más cerradas para que los extremos no queden al borde
        if pref in ("F", "P"):
            return 0.92
    
        # PO ya la tienes bonita, casi no tocar
        if pref == "PO":
            return 1.00
    
        # temporales: mantener, pero si aún chocan bájalo un poco (0.95)
        if pref in ("T", "FT", "TP"):
            return 0.97
    
        return 1.0
    def _proyectar_dentro_circulo(self, x, y, cx, cy, R, margen=0.98):
        """Si un punto cae fuera del círculo, lo proyecta hacia adentro."""
        dx = x - cx
        dy = y - cy
        d2 = dx*dx + dy*dy
        lim2 = (margen*R)*(margen*R)
        if d2 <= lim2 or d2 == 0:
            return x, y
        d = math.sqrt(d2)
        escala = (margen*R) / d
        return cx + dx*escala, cy + dy*escala
    
    def _coord_1010_curvada(self, et: str, cx, cy, R):
        """
        Coordenada 2D basada en:
        - porcentaje Nz->Iz (10-10),
        - lateralidad por número,
        - curvatura opuesta arriba/abajo (como plantilla).
        """
        et = et.upper().strip()
    
        # Mastoides y cerebelo con reglas directas
        # if et == "M1": return (cx - R, cy)
        # if et == "M2": return (cx + R, cy)
        if et == "M1": return (cx - R, cy + 0.18 * R)
        if et == "M2": return (cx + R, cy + 0.18 * R)
        
        if et == "CB1":
            x = cx - 0.43 * R
            y = cy + 0.86 * R
            return self._proyectar_dentro_circulo(x, y, cx, cy, R, margen=0.985)
        
        if et == "CB2":
            x = cx + 0.43 * R
            y = cy + 0.86 * R
            return self._proyectar_dentro_circulo(x, y, cx, cy, R, margen=0.985)
    
        m = re.match(r"^([A-Z]+)(\d+|Z)$", et)
        if not m:
            return None
    
        pref, suf = m.group(1), m.group(2)
    
        # 1) posición antero-posterior (Nz->Iz)
        s = self._s_por_prefijo_1010(pref)          # 0..1
        y_base = cy + (s - 0.5) * 2.0 * R           # Nz arriba (-R), Iz abajo (+R)
        # --- ajuste fino: subir un poco la fila F (frontal)
        if pref == "F":
            y_base -= 0.02 * R   # prueba 0.02 a 0.04
    
        # ancho máximo dentro del círculo a esa altura
        dy = y_base - cy
        x_max = math.sqrt(max(0.0, R*R - dy*dy)) * 0.98
        if x_max < 1:
            return None
    
        # 2) lateralidad
        # 2) lateralidad (mejorada): si la fila tiene orden, usamos espaciado uniforme
        t = self._t_uniforme_en_fila(pref, suf)
        apertura = self._factor_apertura_fila(pref)
        
        # factor general para meter laterales hacia adentro
        # (baja a 0.86-0.92 si los ves muy al borde)
        lateral_global = 0.90
        
        
        # temporales suelen ser más laterales visualmente
        if pref in ("T", "FT", "TP"):
            lateral_global = 0.94
        
        # --- NUEVO: comprimir filas internas para que no choquen con temporales
        if pref in ("C", "FC", "CP"):
            lateral_global = 0.82   # prueba 0.78 a 0.86
        
        if suf == "Z":
            x = cx
            y = y_base
        elif t is not None:
            # espaciado uniforme por fila (F, P, PO)
            x = cx + (t * x_max * lateral_global * apertura)
            y = y_base
        else:
            # fallback: tu método por número
            n = int(suf)
            lf = self._lateral_por_numero(pref, n) * lateral_global
            signo = -1 if (n % 2 == 1) else 1
            x = cx + signo * lf * x_max
            y = y_base
    
        # 3) curvatura opuesta arriba/abajo (tu punto 1)
        # arriba: lados bajan; abajo: lados suben
        if s < 0.50:
            signo_curva = -1
        elif s > 0.50:
            signo_curva = +1
        else:
            signo_curva = 0
    
        # amplitud de curva por fila (F/P un poco más curvados)
        if pref in ("F", "P"):
            amp = 0.14* R
        elif pref in ("FC", "CP", "FT", "TP"):
            amp = 0.10 * R
        elif pref in ("PO", "O", "FP", "AF"):
            amp = 0.08 * R
        else:
            amp = 0.08 * R
    
        xn = (x - cx) / x_max  # -1..1

        # --- ajuste fino: en fila F reducimos un poco la caída de los extremos
        if pref == "F":
            y = y + signo_curva * amp * 0.75 * (xn * xn)   # 0.70 a 0.85
        else:
            y = y + signo_curva * amp * (xn * xn)
    
        # asegurar dentro del círculo
        x, y = self._proyectar_dentro_circulo(x, y, cx, cy, R, margen=0.985)
        return (x, y)
    
    def _dibujar_guia_fila_curvada(self, canvas, cx, cy, R, s, color="#BBB"):
        """Dibuja una 'fila' guía curvada (no un anillo perfecto)."""
        y_base = cy + (s - 0.5) * 2.0 * R
        dy = y_base - cy
        x_max = math.sqrt(max(0.0, R*R - dy*dy)) * 0.985
        if x_max < 5:
            return
    
        # signo de curvatura
        if s < 0.50:
            signo_curva = -1
        elif s > 0.50:
            signo_curva = +1
        else:
            signo_curva = 0
    
        # curva suave
        
        amp = 0.09 * R
        # si esta guía corresponde a la fila F (s=0.30), bajamos un poco la curva
        if abs(s - 0.30) < 1e-6:
            amp = 0.07 * R
        pts = []
        for i in range(101):
            t = -1.0 + 2.0*i/100.0
            x = cx + t * x_max
            y = y_base + signo_curva * amp * (t*t)
            x, y = self._proyectar_dentro_circulo(x, y, cx, cy, R, margen=0.995)
            pts.extend([x, y])
    
        canvas.create_line(*pts, fill=color, width=1, smooth=True)
            
    def _dibujar_mapa_full_en_canvas(self, canvas, scale=1.0):
        canvas.delete("all")
    
        # tamaño visible real
        #vw = max(1, canvas.winfo_width())
        #vh = max(1, canvas.winfo_height())
    
        # lienzo virtual (para zoom general del mapa)
        base_w, base_h = 1400, 1100
        virtual_w = int(base_w * scale)
        virtual_h = int(base_h * scale)
        canvas.configure(scrollregion=(0, 0, virtual_w, virtual_h))
    
        cx, cy = virtual_w // 2, virtual_h // 2
        R = int(min(virtual_w, virtual_h) * 0.40)
    
        # tamaños fijos (electrodos)
        rr_el = 20
        font_el = ("Segoe UI", 9, "bold")
    
        # tamaños relativos
        ear_w = int(0.12 * R)
        ear_h = int(0.18 * R)
        rr_ref = int(0.045 * R)
    
        # ----- cabeza
        canvas.create_oval(cx-R, cy-R, cx+R, cy+R, outline="#555", width=2)
    
        # cruz guía
        canvas.create_line(cx, cy-R, cx, cy+R, fill="#777", width=1)
        canvas.create_line(cx-R, cy, cx+R, cy, fill="#777", width=1)
    
        # nariz
        canvas.create_polygon(cx-14, cy-R+12, cx+14, cy-R+12, cx, cy-R-18,
                              outline="#555", fill="", width=2)
    
        # orejas
        canvas.create_oval(cx-R-ear_w//2, cy-ear_h//2, cx-R+ear_w//2, cy+ear_h//2, outline="#777", width=2)
        canvas.create_oval(cx+R-ear_w//2, cy-ear_h//2, cx+R+ear_w//2, cy+ear_h//2, outline="#777", width=2)
    
        # Nz / Iz
        canvas.create_oval(cx-rr_ref, cy-R-rr_ref*2, cx+rr_ref, cy-R, fill="black", outline="black")
        canvas.create_text(cx, cy-R-rr_ref, text="Nz", fill="white", font=("Segoe UI", 9, "bold"))
    
        canvas.create_oval(cx-rr_ref, cy+R, cx+rr_ref, cy+R+rr_ref*2, fill="black", outline="black")
        canvas.create_text(cx, cy+R+rr_ref, text="Iz", fill="white", font=("Segoe UI", 9, "bold"))
    
        # LPA / RPA decorativo
        canvas.create_oval(cx-R-ear_w, cy-rr_ref, cx-R, cy+rr_ref, fill="black", outline="black")
        canvas.create_text(cx-R-ear_w//2, cy, text="LPA", fill="white", font=("Segoe UI", 8, "bold"))
        canvas.create_oval(cx+R, cy-rr_ref, cx+R+ear_w, cy+rr_ref, fill="black", outline="black")
        canvas.create_text(cx+R+ear_w//2, cy, text="RPA", fill="white", font=("Segoe UI", 8, "bold"))
    
        # disponibles reales (64)
        disponibles = set([n.upper().replace("-AVG", "") for n in self.nombres_canales])

        # canal activo global (sincronizado con el resto de la UI)
        canal_sel = self._nombre_simple(self.var_canal_nombre.get()) if hasattr(self, "var_canal_nombre") else ""

        # guías por filas (curvas) -> NO anillos
        for s, col in [
            (0.10, "#AAA"), (0.20, "#AAA"), (0.30, "#BBB"), (0.40, "#C0C0C0"),
            (0.50, "#999"),
            (0.60, "#C0C0C0"), (0.70, "#BBB"), (0.80, "#AAA"), (0.90, "#AAA"),
        ]:
            self._dibujar_guia_fila_curvada(canvas, cx, cy, R, s, color=col)
    
        # dibujar electrodos
        for et in sorted(disponibles):
            p = self._coord_1010_curvada(et, cx, cy, R)
            if p is None:
                continue
            x, y = p
    
            color, _ = self._color_region(et)
            tag = f"FULL_{et}"
            seleccionado = (self._nombre_simple(et) == canal_sel)

            if seleccionado:
                canvas.create_oval(
                    x - rr_el - 7, y - rr_el - 7, x + rr_el + 7, y + rr_el + 7,
                    fill="#2563EB", outline="white", width=3, tags=(tag,)
                )

            canvas.create_oval(
                x - rr_el, y - rr_el, x + rr_el, y + rr_el,
                fill="white" if seleccionado else color,
                outline="#2563EB" if seleccionado else "#2b2b2b",
                width=2 if seleccionado else 1,
                tags=(tag,)
            )
            canvas.create_text(
                x, y,
                text=et,
                font=font_el,
                fill="#1D4ED8" if seleccionado else "#111827",
                tags=(tag,)
            )
            canvas.tag_bind(tag, "<Button-1>", lambda e, lab=et: self._seleccionar_canal_por_nombre(lab))

    def _dibujar_leyenda_mapa(self, contenedor):
        ley = ttk.Frame(contenedor)
        ley.pack(fill=tk.X, pady=(6, 0))
    
        items = [
            ("Frontal", "#9AD0F5"),
            ("Central/Temporal", "#A7F3D0"),
            ("Parietal", "#C4B5FD"),
            ("Occipital/Cerebelo", "#FCA5A5"),
            ("Mastoides", "#D1D5DB"),
        ]
    
        for txt, col in items:
            fila = ttk.Frame(ley)
            fila.pack(anchor="w")
    
            cuad = tk.Canvas(fila, width=14, height=14, highlightthickness=1, highlightbackground="#999")
            cuad.pack(side=tk.LEFT, padx=(0, 6))
            cuad.create_rectangle(0, 0, 14, 14, fill=col, outline=col)
    
            ttk.Label(fila, text=txt).pack(side=tk.LEFT)

    def _on_configure_mapa_regional(self, event=None):
        if self._cerrando:
            return
    
        if not hasattr(self, "canvas_mapa") or self.canvas_mapa is None:
            return
    
        if not self.canvas_mapa.winfo_exists():
            return
    
        # evitar dibujar cuando el canvas aún no tiene tamaño real
        w = self.canvas_mapa.winfo_width()
        h = self.canvas_mapa.winfo_height()
        if w < 20 or h < 20:
            return
    
        # debounce simple para evitar redibujos seguidos
        if hasattr(self, "_after_mapa_regional") and self._after_mapa_regional is not None:
            try:
                self.raiz.after_cancel(self._after_mapa_regional)
            except Exception:
                pass
    
        self._after_mapa_regional = self.raiz.after(80, self._dibujar_mapa_regional_resumen)
    

    def _limitar_bandas(self):
        seleccionadas = [b for b in self.bandas if self.band_vars[b].get()]
        if len(seleccionadas) > 4:
            for b in reversed(self.bandas):
                if self.band_vars[b].get():
                    self.band_vars[b].set(False)
                    break
            messagebox.showwarning("Máximo", "Máximo 4 bandas al tiempo.")
        seleccionadas = [b for b in self.bandas if self.band_vars[b].get()]
        if len(seleccionadas) == 0:
            self.band_vars["Alfa"].set(True)

    ## =========================================================
    ## Flujo principal: seleccionar, procesar e informe
    ## =========================================================
    def _copiar_ruta_carpeta(self):
        ruta = getattr(self, "carpeta_dcl", None)
        if not ruta:
            return
        self.raiz.clipboard_clear()
        self.raiz.clipboard_append(ruta)
        self._set_barra("Ruta copiada al portapapeles.")

    def _abrir_carpeta_en_explorador(self):
        ruta = getattr(self, "carpeta_dcl", None)
        if not ruta or not os.path.isdir(ruta):
            self._set_barra("No hay carpeta seleccionada.")
            return
        try:
            subprocess.Popen(["explorer", os.path.normpath(ruta)])
        except Exception as e:
            self._set_barra(f"No se pudo abrir el explorador: {e}")

    def seleccionar_carpeta(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta con los archivos .dat")
        if not carpeta:
            return
    
        self.carpeta_dcl = carpeta
        self.lbl_carpeta.config(text=f"Carpeta actual:\n{self._formatear_ruta_panel(self.carpeta_dcl)}")
        self.guardar_config()
    
        try:
            archivos = [f for f in os.listdir(self.carpeta_dcl) if f.lower().endswith(".dat")]
        except Exception as e:
            messagebox.showerror("Error", f"No pude leer la carpeta:\n{e}")
            return
    
        archivos.sort()
        self.archivos_dat = archivos
    
        self.lista.delete(*self.lista.get_children())
        for f in archivos:
            self.lista.insert("", tk.END, iid=f, text=f, tags=("noprocesado",))

        if not archivos:
            self.var_resumen_lista.set("0 archivos .dat")
            self._set_barra("No encontré archivos .dat en esa carpeta.")
            self.log("No encontré archivos .dat en esa carpeta.")
        else:
            self.var_resumen_lista.set(f"{len(archivos)} archivo(s) .dat")
            self._set_barra(f"Listo: {len(archivos)} archivo(s) .dat encontrado(s).")
            self.log(f"Carpeta cargada: {self.carpeta_dcl}")
            self.log(f"Encontré {len(archivos)} archivo(s) .dat")
        self.raiz.after_idle(self._actualizar_colores_lista)

    def on_select_dat(self, _evt=None):
        idxs = self.lista.selection()
        if not idxs:
            return
        nombre = idxs[0]  # iid == nombre de archivo
        self.var_resumen_lista.set(f"Seleccionado: {nombre}")
        carpeta_cache = self._ruta_cache_de_archivo(nombre)
        if self._cache_valido(carpeta_cache):
            self._usar_cache_existente(nombre, carpeta_cache)
        else:
            self.cache_ultimo = None
            self.nombre_ultimo = nombre
            self.ruta_pdf_ultimo = None
            self.btn_proc.config(state="normal")
            self.btn_pdf.config(state="disabled")
            self._actualizar_bloques_resumen_calidad()
            self._activar_tab_global("grafica")
            self._mostrar_marco_visor_grafica(False)
            self._mostrar_tabs_grafica(False)
            self._mostrar_placeholder_visor(
                f"'{nombre}' no tiene datos procesados.\nUsa 'Procesar' para procesar este archivo."
            )
            self._set_barra(f"Sin datos: {nombre}. Usa 'Procesar'.")
            self.log(f"Seleccionado: {nombre}")
            self.log("[Aviso] No hay datos procesados. Usa 'Procesar'.")

    def _ruta_cache_de_archivo(self, nombre_archivo):
        nombre_limpio = os.path.splitext(os.path.basename(nombre_archivo))[0]
        nombre_limpio = nombre_limpio.replace(" ", "_").replace(".", "_")
        return os.path.join(self.carpeta_dcl, "CACHE", nombre_limpio)

    def _toggle_sidebar(self):
        if self._sidebar_visible:
            self._cont_izq.pack_forget()
            self._btn_toggle_sidebar.config(text="▶")
            self._sidebar_visible = False
        else:
            self._cont_izq.pack(side=tk.LEFT, fill=tk.Y, before=self._cont_der)
            self._btn_toggle_sidebar.config(text="◀")
            self._sidebar_visible = True
        self.raiz.after_idle(self._actualizar_scrollregion_izquierda)

    def _actualizar_colores_lista(self):
        """Procesados → negrita. Sin procesar → gris claro."""
        for nombre in getattr(self, "archivos_dat", []):
            carpeta = self._ruta_cache_de_archivo(nombre)
            tag = "procesado" if self._cache_valido(carpeta) else "noprocesado"
            try:
                self.lista.item(nombre, tags=(tag,))
            except Exception:
                pass

    def _nombre_pdf_informe(self, nombre_archivo):
        nombre_limpio = os.path.splitext(os.path.basename(nombre_archivo))[0]
        nombre_limpio = nombre_limpio.replace(" ", "_").replace(".", "_")
        return f"Informe_{nombre_limpio}.pdf"

    def _ruta_pdf_externo(self, nombre_archivo):
        if not self.carpeta_dcl:
            return None
        return os.path.join(self.carpeta_dcl, "INFORMES_NEUROX", self._nombre_pdf_informe(nombre_archivo))

    def _buscar_pdf_preferido(self, nombre_archivo, carpeta_cache):
        ruta_externa = self._ruta_pdf_externo(nombre_archivo)
        if ruta_externa and os.path.exists(ruta_externa):
            return os.path.abspath(ruta_externa)
        return self._buscar_pdf_en_cache(carpeta_cache)
    
    
    def _cache_valido(self, carpeta_cache):
        if not carpeta_cache or not os.path.isdir(carpeta_cache):
            return False
    
        rutas_minimas = [
            os.path.join(carpeta_cache, "filtrado", "eeg_filtrado.npy"),
            os.path.join(carpeta_cache, "crudo", "tiempo.npy"),
        ]
        return all(os.path.exists(r) for r in rutas_minimas)
    
    
    def _buscar_pdf_en_cache(self, carpeta_cache):
        import glob
    
        if not carpeta_cache or not os.path.isdir(carpeta_cache):
            return None
    
        carpeta_informe = os.path.join(carpeta_cache, "informe")
        candidatos = []
    
        if os.path.isdir(carpeta_informe):
            candidatos += glob.glob(os.path.join(carpeta_informe, "*.pdf"))
            candidatos += glob.glob(os.path.join(carpeta_informe, "*.PDF"))
            candidatos += glob.glob(os.path.join(carpeta_informe, "**", "*.pdf"), recursive=True)
            candidatos += glob.glob(os.path.join(carpeta_informe, "**", "*.PDF"), recursive=True)
    
        if not candidatos:
            candidatos += glob.glob(os.path.join(carpeta_cache, "**", "*.pdf"), recursive=True)
            candidatos += glob.glob(os.path.join(carpeta_cache, "**", "*.PDF"), recursive=True)
    
        if not candidatos:
            return None
    
        candidatos = [os.path.abspath(p) for p in candidatos if os.path.exists(p)]
        if not candidatos:
            return None
    
        candidatos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidatos[0]
    
    
    def _usar_cache_existente(self, nombre_archivo, carpeta_cache):
        # ── Capturar estado ANTES de cualquier modificación ──────────────
        _primer_archivo  = not bool(self.cache_ultimo)
        _en_analisis     = (
            not _primer_archivo
            and hasattr(self, "panel_analisis")
            and self.panel_analisis.winfo_ismapped()
        )
        _modo_previo     = self.var_modo.get() if hasattr(self, "var_modo") else ""
        _subtab_grafica  = self.tabs_grafica.select()  if hasattr(self, "tabs_grafica")  else None
        _subtab_analisis = self.tabs_analisis.select() if hasattr(self, "tabs_analisis") else None
        # ─────────────────────────────────────────────────────────────────

        if hasattr(self, "_loading"):
            self._loading.show()

        self._liberar_cache_visual()

        self.cache_ultimo = carpeta_cache
        self.nombre_ultimo = nombre_archivo

        ruta_pdf = self._buscar_pdf_preferido(nombre_archivo, carpeta_cache)
        self.ruta_pdf_ultimo = ruta_pdf

        if ruta_pdf and os.path.exists(ruta_pdf):
            self.btn_pdf.config(state="normal")
        else:
            self.btn_pdf.config(state="disabled")

        self.btn_proc.config(state="normal")
        self._actualizar_canales_alerta()
        self._actualizar_bloques_resumen_calidad()
        self._er_filtrado = None

        self._set_barra(f"Usando cache existente: {nombre_archivo}")
        self.log("====================================")
        self.log(f"Usando cache existente para: {nombre_archivo}")
        if ruta_pdf:
            self.log("[OK] Informe localizado para este registro.")
        else:
            self.log("[Aviso] No encontré informe PDF, pero sí datos para visualizar.")

        self._reset_panel_analisis()
        self._er_filtrado = None

        if _en_analisis:
            # Quedarse en Análisis: sin pack/unpack de paneles → sin flash visual
            if _subtab_analisis:
                try:
                    self.tabs_analisis.select(_subtab_analisis)
                except Exception:
                    pass
            # El evento <<NotebookTabChanged>> no se dispara si el tab ya estaba seleccionado.
            # Llamamos _on_tab_analisis directamente para recargar el contenido del sub-tab activo.
            self.raiz.after(50, self._on_tab_analisis)
        else:
            # Primera carga o estaba en Gráficas → ir a Gráficas
            self._activar_tab_global("grafica")
            self._activar_grafica_inicial()
            if not _primer_archivo and _modo_previo and _modo_previo != "Filtrado multicanal":
                try:
                    self.var_modo.set(_modo_previo)
                    if _subtab_grafica:
                        self.tabs_grafica.select(_subtab_grafica)
                    self._refrescar_ui_modo()
                    self.raiz.after(80, self.graficar)
                except Exception:
                    pass

        self.log("[OK] Resultados actualizados en la interfaz.")
        if hasattr(self, "_loading"):
            self.raiz.after(150, self._loading.hide)

    def procesar_seleccionado(self):
        if not self.carpeta_dcl or not os.path.isdir(self.carpeta_dcl):
            messagebox.showwarning(
                "Falta carpeta",
                "Selecciona primero una carpeta válida con archivos .dat/.dap."
            )
            return
    
        idxs = self.lista.selection()
        if not idxs:
            messagebox.showwarning("Sin selección", "Selecciona un archivo .dat de la lista.")
            return

        nombre_archivo = idxs[0]  # iid == nombre de archivo
        ruta_archivo = os.path.join(self.carpeta_dcl, nombre_archivo)


        if not os.path.exists(ruta_archivo):
            messagebox.showerror("No existe", f"No encuentro el archivo:\n{ruta_archivo}")
            return
    
        carpeta_cache = self._ruta_cache_de_archivo(nombre_archivo)
    
        # Si ya existe cache, preguntar qué hacer
        if self._cache_valido(carpeta_cache):
            resp = messagebox.askyesnocancel(
                "Archivo ya procesado",
                "Este archivo ya tiene datos procesados.\n\n"
                "Sí  = volver a procesar\n"
                "No  = solo visualizar lo ya guardado\n"
                "Cancelar = no hacer nada"
            )
    
            if resp is None:
                return
    
            # No = solo visualizar
            if resp is False:
                self._usar_cache_existente(nombre_archivo, carpeta_cache)
                return
    
            # Sí = Procesar
            self.limpiar_log()
            self.log("====================================")
            self.log(f"Reprocesando: {nombre_archivo}")
    
        else:
            self.limpiar_log()
            self.log("====================================")
            self.log(f"Procesando: {nombre_archivo}")
    
        self._liberar_cache_visual()
    
        self._set_barra(f"Procesando: {nombre_archivo} ...")
        self._iniciar_trabajo("Preparando procesamiento...")
    
        th = threading.Thread(target=self._hilo_procesar, args=(nombre_archivo,), daemon=True)
        th.start()
        

    def _hilo_procesar(self, nombre_archivo):
        carpeta_cache = None
        err = None

        def logger(msg):
            if self._cerrando:
                return
            try:
                self.raiz.after(0, lambda: self.log(str(msg)))
            except Exception:
                pass

        def progress_cb(valor, mensaje=None):
            if self._cerrando:
                return
            try:
                self.raiz.after(0, lambda: self.set_progreso(valor, mensaje))
            except Exception:
                pass

        try:
            carpeta_cache = procesar_archivo(
                nombre_archivo,
                carpeta_base=self.carpeta_dcl,
                logger=logger,
                progress_callback=progress_cb
            )
        except Exception as e:
            err = e

        def done_proceso():
            if self._cerrando:
                return

            if err:
                self.finalizar_progreso(
                    "Error durante el procesamiento.",
                    valor=self.progress_var.get(),
                    delay_ms=2200
                )
                self._set_barra("Error en procesamiento.")
                self.log(f"[Error] No se pudo procesar el archivo: {err}")
                messagebox.showerror("Error", f"No se pudo procesar el archivo.\n\n{err}")
                return

            if not carpeta_cache:
                self.finalizar_progreso(
                    "Error durante el procesamiento.",
                    valor=self.progress_var.get(),
                    delay_ms=2200
                )
                self._set_barra("No se pudo completar el procesamiento.")
                self.log("[Error] El pipeline no devolvio la ruta de cache.")
                return

            self.cache_ultimo = carpeta_cache
            self.nombre_ultimo = nombre_archivo
            self._set_barra(f"Procesado OK: {nombre_archivo}")
            self.log("[OK] Datos procesados y guardados en cache.")
            self.set_progreso(95, "Generando informe PDF...")
            self.log("[Proceso] Generando informe PDF...")
            self._set_barra("Generando informe PDF...")

            th2 = threading.Thread(target=self._hilo_informe, daemon=True)
            th2.start()

        try:
            self.raiz.after(0, done_proceso)
        except Exception:
            pass
    

    def _hilo_informe(self):
        err = None
        ruta_pdf = None

        def logger(msg):
            if self._cerrando:
                return
            try:
                self.raiz.after(0, lambda: self.log(str(msg)))
            except Exception:
                pass

        try:
            try:
                self.raiz.after(0, lambda: self.set_progreso(95, "Generando informe PDF..."))
            except Exception:
                pass
            ruta_pdf = generar_informe_desde_cache(self.cache_ultimo, logger=logger)
        except Exception as e:
            err = e

        def done_informe():
            if self._cerrando:
                return
            if err:
                self.finalizar_progreso(
                    "Error durante el procesamiento.",
                    valor=self.progress_var.get(),
                    delay_ms=2200
                )
                self._set_barra("Error generando informe.")
                self.log(f"[Error] No se pudo generar el informe: {err}")
                messagebox.showerror("Error", f"No se pudo generar el informe.\n\n{err}")
                return

            # (tabs siempre activos — sin estado disabled en el nuevo diseño)

            if ruta_pdf and os.path.exists(ruta_pdf):
                self.ruta_pdf_ultimo = os.path.abspath(ruta_pdf)
                self.btn_pdf.config(state="normal")
                self._liberar_cache_visual()
                self._er_filtrado = None
                self._limpiar_estado_welch()
                self._reset_panel_analisis()
                self._actualizar_canales_alerta()
                self._actualizar_bloques_resumen_calidad()
                self._activar_grafica_inicial()
                self.raiz.after_idle(self._actualizar_colores_lista)
                self.finalizar_progreso("Procesamiento finalizado.", valor=100.0, delay_ms=1800)
                self._set_barra(f"Informe listo: {os.path.basename(ruta_pdf)}")
                self.log("[OK] Informe PDF generado correctamente.")
                self.log("[OK] Resultados actualizados en la interfaz.")

                # Ya NO se abre el PDF automáticamente. Solo se avisa que
                # terminó bien; para verlo, el usuario usa "Vista previa".
                messagebox.showinfo(
                    "Procesado con éxito",
                    f"El archivo se procesó correctamente y el informe está listo.\n\n"
                    f"Usa 'Vista previa' para verlo cuando quieras."
                )
            else:
                self.ruta_pdf_ultimo = self._buscar_pdf_preferido(self.nombre_ultimo, self.cache_ultimo)
                self.btn_pdf.config(state="normal" if self.ruta_pdf_ultimo and os.path.exists(self.ruta_pdf_ultimo) else "disabled")
                self._liberar_cache_visual()
                self._reset_panel_analisis()
                self._actualizar_canales_alerta()
                self._actualizar_bloques_resumen_calidad()
                self._activar_grafica_inicial()
                self.finalizar_progreso("Procesamiento finalizado.", valor=100.0, delay_ms=1800)
                self._set_barra("Procesamiento finalizado, pero no se encontró el informe PDF.")
                self.log("[Aviso] Se generó el procesamiento, pero no se encontró el informe PDF.")
                self.log("[OK] Resultados actualizados en la interfaz.")
                messagebox.showwarning(
                    "Informe",
                    "El procesamiento terminó, pero no aparece ningún PDF disponible.\n"
                    "Revisa el log para más detalle."
                )

        try:
            if not self._cerrando:
                self.raiz.after(0, done_informe)
        except Exception:
            pass

    def abrir_informe(self):
        if (not self.ruta_pdf_ultimo or not os.path.exists(self.ruta_pdf_ultimo)) and self.nombre_ultimo and self.cache_ultimo:
            self.ruta_pdf_ultimo = self._buscar_pdf_preferido(self.nombre_ultimo, self.cache_ultimo)

        if not self.ruta_pdf_ultimo or not os.path.exists(self.ruta_pdf_ultimo):
            messagebox.showwarning("Informe", "Todavía no hay informe generado.")
            return
        try:
            abrir_pdf_en_windows(self.ruta_pdf_ultimo)
            self.log("[OK] Informe abierto en el visor de Windows.")
        except Exception as e:
            messagebox.showerror("PDF", f"No pude abrir el PDF:\n{e}")

    ## =========================================================
    ## Consolidado Excel (varios informes en un solo .xlsx)
    ## =========================================================
    def abrir_ventana_consolidado(self):
        """
        Ventana NO modal (sin grab_set) para elegir qué informes consolidar.
        El usuario puede cerrarla en cualquier momento con 'Cerrar' o con la (X)
        y seguir viendo/gráficando otros archivos en la ventana principal mientras
        tanto, incluso si dejó una generación corriendo en segundo plano.
        """
        if not self.carpeta_dcl or not getattr(self, "archivos_dat", None):
            messagebox.showinfo("Consolidado", "Primero selecciona una carpeta con archivos .dat.")
            return

        disponibles = []
        for nombre in self.archivos_dat:
            carpeta_cache = self._ruta_cache_de_archivo(nombre)
            if self._cache_valido(carpeta_cache):
                disponibles.append((nombre, carpeta_cache))

        if not disponibles:
            messagebox.showinfo(
                "Consolidado",
                "Todavía no hay archivos procesados.\nProcesa al menos un archivo antes de generar el consolidado."
            )
            return

        win = tk.Toplevel(self.raiz)
        win.title("Descargar consolidado Excel")
        win.geometry("440x520")
        win.configure(bg=FONDO_APP)
        win.transient(self.raiz)   # queda asociada a la ventana principal...
        win.attributes("-topmost", False)
        # ... pero SIN grab_set(): no es modal, así que el usuario puede seguir
        # usando/viendo otros informes en la ventana principal mientras esta sigue abierta.

        ttk.Label(
            win,
            text="Selecciona los informes que quieres incluir en el consolidado:",
            style="Sidebar.TLabel",
            wraplength=400,
            justify="left",
        ).pack(fill=tk.X, padx=12, pady=(12, 6))

        marco_lista = ttk.Frame(win, style="Sidebar.TFrame")
        marco_lista.pack(fill=tk.BOTH, expand=True, padx=12)

        canvas_sel = tk.Canvas(marco_lista, bg=FONDO_APP, highlightthickness=0)
        scroll_sel = ttk.Scrollbar(
            marco_lista, orient="vertical", command=canvas_sel.yview, style="NeuroX.Vertical.TScrollbar"
        )
        frame_check = ttk.Frame(canvas_sel, style="Sidebar.TFrame")
        frame_check.bind(
            "<Configure>", lambda e: canvas_sel.configure(scrollregion=canvas_sel.bbox("all"))
        )
        canvas_sel.create_window((0, 0), window=frame_check, anchor="nw")
        canvas_sel.configure(yscrollcommand=scroll_sel.set)
        canvas_sel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_sel.pack(side=tk.RIGHT, fill=tk.Y)

        variables = {}
        for nombre, carpeta_cache in disponibles:
            var = tk.BooleanVar(value=True)
            variables[nombre] = (var, carpeta_cache)
            ttk.Checkbutton(frame_check, text=nombre, variable=var).pack(anchor="w", pady=2, padx=2)

        frame_todos = ttk.Frame(win, style="Sidebar.TFrame")
        frame_todos.pack(fill=tk.X, padx=12, pady=(6, 0))

        def _marcar_todos(valor):
            for var, _ in variables.values():
                var.set(valor)

        ttk.Button(
            frame_todos, text="Marcar todos", style="Secondary.TButton",
            command=lambda: _marcar_todos(True)
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            frame_todos, text="Desmarcar todos", style="Secondary.TButton",
            command=lambda: _marcar_todos(False)
        ).pack(side=tk.LEFT)

        var_estado_consolidado = tk.StringVar(value="")
        ttk.Label(
            win, textvariable=var_estado_consolidado, style="Sidebar.TLabel",
            wraplength=400, justify="left"
        ).pack(fill=tk.X, padx=12, pady=(8, 0))

        frame_final = ttk.Frame(win, style="Sidebar.TFrame")
        frame_final.pack(fill=tk.X, padx=12, pady=12)

        btn_generar = ttk.Button(frame_final, text="Generar Excel", style="Accent.TButton")
        btn_generar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        # Botón 'Cerrar': cierra SOLO esta ventana; la ventana principal sigue
        # activa y el usuario puede seguir viendo/abriendo otros informes.
        ttk.Button(
            frame_final, text="Cerrar", style="Secondary.TButton", command=win.destroy
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 0))

        def _generar():
            seleccionados = [
                (nombre, carpeta_cache)
                for nombre, (var, carpeta_cache) in variables.items()
                if var.get()
            ]
            if not seleccionados:
                messagebox.showwarning("Consolidado", "Selecciona al menos un archivo.", parent=win)
                return

            ruta_salida = filedialog.asksaveasfilename(
                parent=win,
                title="Guardar consolidado como",
                defaultextension=".xlsx",
                filetypes=[("Libro de Excel", "*.xlsx")],
                initialfile="Consolidado_reportes_neuroX.xlsx",
            )
            if not ruta_salida:
                return

            btn_generar.config(state="disabled")
            var_estado_consolidado.set(f"Generando consolidado con {len(seleccionados)} informe(s)...")

            def _hilo_consolidado():
                err = None
                incluidos, omitidos = [], []
                try:
                    _, incluidos, omitidos = generar_consolidado_excel(
                        seleccionados,
                        ruta_salida,
                        logger=lambda m: self.raiz.after(0, lambda: self.log(str(m))),
                    )
                except Exception as e:
                    err = e

                def _terminar():
                    if not win.winfo_exists():
                        # El usuario cerró la ventana de selección mientras se generaba:
                        # igual avisamos por el log/barra de estado de la ventana principal.
                        if err:
                            self.log(f"[Error] No se pudo generar el consolidado: {err}")
                            self._set_barra("Error generando el consolidado.")
                        else:
                            self.log(f"[OK] Consolidado generado en: {ruta_salida}")
                            self._set_barra("Consolidado Excel generado.")
                        return

                    btn_generar.config(state="normal")
                    if err:
                        var_estado_consolidado.set(f"Error: {err}")
                        messagebox.showerror(
                            "Consolidado", f"No se pudo generar el consolidado.\n\n{err}", parent=win
                        )
                        return

                    mensaje = f"Consolidado generado: {os.path.basename(ruta_salida)}"
                    if omitidos:
                        mensaje += f"\nOmitidos (sin datos procesados completos): {', '.join(omitidos)}"
                    var_estado_consolidado.set(mensaje)
                    self.log(f"[OK] Consolidado Excel generado en: {ruta_salida}")
                    self._set_barra("Consolidado Excel generado.")

                try:
                    self.raiz.after(0, _terminar)
                except Exception:
                    pass

            threading.Thread(target=_hilo_consolidado, daemon=True).start()

        btn_generar.config(command=_generar)

    ## =========================================================
    ## Consolidado PDF (varios informes PDF combinados en uno solo)
    ## =========================================================
    def abrir_ventana_consolidado_pdf(self):
        """
        Igual que la ventana del consolidado Excel: NO modal (sin grab_set),
        con botón 'Cerrar' independiente, para poder seguir viendo otros
        informes en la ventana principal mientras esta sigue abierta.
        """
        if not self.carpeta_dcl or not getattr(self, "archivos_dat", None):
            messagebox.showinfo("Consolidado PDF", "Primero selecciona una carpeta con archivos .dat.")
            return

        disponibles = []
        for nombre in self.archivos_dat:
            carpeta_cache = self._ruta_cache_de_archivo(nombre)
            if not self._cache_valido(carpeta_cache):
                continue
            ruta_pdf = self._buscar_pdf_preferido(nombre, carpeta_cache)
            if ruta_pdf and os.path.exists(ruta_pdf):
                disponibles.append((nombre, ruta_pdf))

        if not disponibles:
            messagebox.showinfo(
                "Consolidado PDF",
                "Todavía no hay informes PDF generados.\nProcesa al menos un archivo antes de generar el consolidado."
            )
            return

        win = tk.Toplevel(self.raiz)
        win.title("Descargar consolidado PDF")
        win.geometry("440x520")
        win.configure(bg=FONDO_APP)
        win.transient(self.raiz)  # asociada a la ventana principal, pero NO modal (sin grab_set)

        ttk.Label(
            win,
            text="Selecciona los informes PDF que quieres combinar en un solo archivo:",
            style="Sidebar.TLabel",
            wraplength=400,
            justify="left",
        ).pack(fill=tk.X, padx=12, pady=(12, 6))

        marco_lista = ttk.Frame(win, style="Sidebar.TFrame")
        marco_lista.pack(fill=tk.BOTH, expand=True, padx=12)

        canvas_sel = tk.Canvas(marco_lista, bg=FONDO_APP, highlightthickness=0)
        scroll_sel = ttk.Scrollbar(
            marco_lista, orient="vertical", command=canvas_sel.yview, style="NeuroX.Vertical.TScrollbar"
        )
        frame_check = ttk.Frame(canvas_sel, style="Sidebar.TFrame")
        frame_check.bind(
            "<Configure>", lambda e: canvas_sel.configure(scrollregion=canvas_sel.bbox("all"))
        )
        canvas_sel.create_window((0, 0), window=frame_check, anchor="nw")
        canvas_sel.configure(yscrollcommand=scroll_sel.set)
        canvas_sel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_sel.pack(side=tk.RIGHT, fill=tk.Y)

        variables = {}
        for nombre, ruta_pdf in disponibles:
            var = tk.BooleanVar(value=True)
            variables[nombre] = (var, ruta_pdf)
            ttk.Checkbutton(frame_check, text=nombre, variable=var).pack(anchor="w", pady=2, padx=2)

        frame_todos = ttk.Frame(win, style="Sidebar.TFrame")
        frame_todos.pack(fill=tk.X, padx=12, pady=(6, 0))

        def _marcar_todos(valor):
            for var, _ in variables.values():
                var.set(valor)

        ttk.Button(
            frame_todos, text="Marcar todos", style="Secondary.TButton",
            command=lambda: _marcar_todos(True)
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            frame_todos, text="Desmarcar todos", style="Secondary.TButton",
            command=lambda: _marcar_todos(False)
        ).pack(side=tk.LEFT)

        var_estado_pdf = tk.StringVar(value="")
        ttk.Label(
            win, textvariable=var_estado_pdf, style="Sidebar.TLabel",
            wraplength=400, justify="left"
        ).pack(fill=tk.X, padx=12, pady=(8, 0))

        frame_final = ttk.Frame(win, style="Sidebar.TFrame")
        frame_final.pack(fill=tk.X, padx=12, pady=12)

        btn_generar = ttk.Button(frame_final, text="Generar PDF", style="Accent.TButton")
        btn_generar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        # 'Cerrar' solo cierra esta ventana; la principal sigue disponible.
        ttk.Button(
            frame_final, text="Cerrar", style="Secondary.TButton", command=win.destroy
        ).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 0))

        def _generar():
            seleccionados = [
                (nombre, ruta_pdf)
                for nombre, (var, ruta_pdf) in variables.items()
                if var.get()
            ]
            if not seleccionados:
                messagebox.showwarning("Consolidado PDF", "Selecciona al menos un informe.", parent=win)
                return

            ruta_salida = filedialog.asksaveasfilename(
                parent=win,
                title="Guardar consolidado PDF como",
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                initialfile="Consolidado_informes_neuroX.pdf",
            )
            if not ruta_salida:
                return

            btn_generar.config(state="disabled")
            var_estado_pdf.set(f"Combinando {len(seleccionados)} informe(s)...")

            def _hilo_pdf():
                err = None
                incluidos, omitidos = [], []
                try:
                    rutas_pdf = [ruta for _, ruta in seleccionados]
                    _, incluidos, omitidos = combinar_informes_pdf(
                        rutas_pdf,
                        ruta_salida,
                        logger=lambda m: self.raiz.after(0, lambda: self.log(str(m))),
                    )
                except Exception as e:
                    err = e

                def _terminar():
                    if not win.winfo_exists():
                        if err:
                            self.log(f"[Error] No se pudo generar el consolidado PDF: {err}")
                            self._set_barra("Error generando el consolidado PDF.")
                        else:
                            self.log(f"[OK] Consolidado PDF generado en: {ruta_salida}")
                            self._set_barra("Consolidado PDF generado.")
                        return

                    btn_generar.config(state="normal")
                    if err:
                        var_estado_pdf.set(f"Error: {err}")
                        messagebox.showerror(
                            "Consolidado PDF", f"No se pudo generar el consolidado.\n\n{err}", parent=win
                        )
                        return

                    mensaje = f"Consolidado PDF generado: {os.path.basename(ruta_salida)}"
                    if omitidos:
                        mensaje += f"\nOmitidos (no se pudo leer el PDF): {len(omitidos)}"
                    var_estado_pdf.set(mensaje)
                    self.log(f"[OK] Consolidado PDF generado en: {ruta_salida}")
                    self._set_barra("Consolidado PDF generado.")

                try:
                    self.raiz.after(0, _terminar)
                except Exception:
                    pass

            threading.Thread(target=_hilo_pdf, daemon=True).start()

        btn_generar.config(command=_generar)

    ## =========================================================
    ## Navegación entre visor y análisis
    ## =========================================================
    def limpiar_panel_izquierdo_contextual(self):
        """
        Oculta todos los paneles variables de la barra izquierda.
        No destruye widgets: preserva callbacks y referencias ya creadas.
        """
        for panel in (
            getattr(self, "panel_izq_grafica", None),
            getattr(self, "panel_izq_analisis", None),
        ):
            if panel is not None:
                try:
                    panel.pack_forget()
                except Exception:
                    pass

        self._ocultar_subpaneles_analisis()

        try:
            self._tooltip_ocultar()
        except Exception:
            pass

    def _ocultar_subpaneles_analisis(self):
        for frame in (
            getattr(self, "frm_energia_relativa_izq", None),
            getattr(self, "frm_potencia_welch_izq", None),
            getattr(self, "frm_ratios_bandas_izq", None),
            getattr(self, "frm_calidad_senal_izq", None),
        ):
            if frame is not None:
                try:
                    frame.pack_forget()
                except Exception:
                    pass

    def mostrar_panel_contextual_grafica(self):
        self.limpiar_panel_izquierdo_contextual()
        self.panel_izq_grafica.pack(fill=tk.X)
        self._actualizar_scrollregion_izquierda()

    def mostrar_panel_contextual_analisis(self):
        self.limpiar_panel_izquierdo_contextual()
        self.panel_izq_analisis.pack(fill=tk.X)
        self._mostrar_panel_izq_analisis_actual()
        self._actualizar_scrollregion_izquierda()

    def mostrar_panel_contextual_energia_relativa(self):
        self._ocultar_subpaneles_analisis()
        self._mostrar_resumen_calidad_compacto_analisis()
        if hasattr(self, "frm_energia_relativa_izq"):
            self.frm_energia_relativa_izq.pack(fill=tk.X)
        self._actualizar_scrollregion_izquierda()

    def mostrar_panel_contextual_potencia_relativa(self):
        self._ocultar_subpaneles_analisis()
        self._mostrar_resumen_calidad_compacto_analisis()
        if hasattr(self, "frm_potencia_welch_izq"):
            self.frm_potencia_welch_izq.pack(fill=tk.X)
        self._actualizar_scrollregion_izquierda()

    def mostrar_panel_contextual_ratios_bandas(self):
        self._ocultar_subpaneles_analisis()
        self._mostrar_resumen_calidad_compacto_analisis()
        if hasattr(self, "frm_ratios_bandas_izq"):
            self.frm_ratios_bandas_izq.pack(fill=tk.X)
        self._actualizar_scrollregion_izquierda()

    def mostrar_panel_contextual_calidad(self):
        self._ocultar_subpaneles_analisis()
        if hasattr(self, "frm_resumen_calidad_analisis"):
            self.frm_resumen_calidad_analisis.pack_forget()
        if hasattr(self, "frm_calidad_senal_izq"):
            self.frm_calidad_senal_izq.pack(fill=tk.X)
        self._actualizar_scrollregion_izquierda()

    def _mostrar_panel_izquierdo(self, modo):
        if modo == "grafica":
            self.mostrar_panel_contextual_grafica()
        elif modo == "analisis":
            self.mostrar_panel_contextual_analisis()
        else:
            self.limpiar_panel_izquierdo_contextual()

    def _activar_tab_global(self, tab):
        if hasattr(self, "tabbar_global"):
            self.tabbar_global.seleccionar(0 if tab == "grafica" else 1)
        if tab == "grafica":
            self.panel_analisis.pack_forget()
            self.panel_grafica.pack(fill=tk.BOTH, expand=True)
            self._actualizar_bloques_resumen_calidad()
            self._mostrar_marco_visor_grafica(bool(self.cache_ultimo))
            self._mostrar_tabs_grafica(bool(self.cache_ultimo))
            self._sincronizar_tab_grafica()
            self._mostrar_panel_izquierdo("grafica")
        else:
            self._mostrar_marco_visor_grafica(False)
            self._mostrar_tabs_grafica(False)
            self.panel_grafica.pack_forget()
            self.panel_analisis.pack(fill=tk.BOTH, expand=True)
            self._actualizar_bloques_resumen_calidad()
            self._mostrar_panel_izquierdo("analisis")
            self._seleccionar_tab_energia_relativa()
            self._reset_panel_analisis()
            self.raiz.update_idletasks()
            self._programar_redibujo_welch()
            self.raiz.after(120, self._programar_redibujo_welch)
            self.raiz.after(350, self._programar_redibujo_welch)
            if getattr(self, "hay_grafico_activo", False) and self.canal_activo_simple:
                self._click_canal_analisis(self.canal_activo_simple)
            self._actualizar_barra_modo_analisis("welch")

    def _reprocesar_forzado(self):
        if not self.carpeta_dcl or not os.path.isdir(self.carpeta_dcl):
            messagebox.showwarning("Falta carpeta", "Selecciona primero una carpeta con archivos .dat.")
            return

        idxs = list(self.lista.selection())
        if not idxs:
            messagebox.showwarning(
                "Sin selección",
                "Selecciona uno o más archivos .dat de la lista.\n\n"
                "Tip: usa Ctrl+clic o Shift+clic para elegir varios."
            )
            return

        # Validar que existan en disco y separar los que ya tienen caché
        nombres_validos = []
        for nombre_archivo in idxs:
            ruta_archivo = os.path.join(self.carpeta_dcl, nombre_archivo)
            if not os.path.exists(ruta_archivo):
                messagebox.showerror("No existe", f"No encuentro el archivo:\n{ruta_archivo}")
                return
            nombres_validos.append(nombre_archivo)

        ya_procesados = [
            n for n in nombres_validos if self._cache_valido(self._ruta_cache_de_archivo(n))
        ]

        cola = list(nombres_validos)
        if ya_procesados:
            if len(nombres_validos) == 1:
                resp = messagebox.askyesno(
                    "Archivo ya procesado",
                    f"'{nombres_validos[0]}' ya tiene datos procesados.\n\n¿Desea reprocesarlo nuevamente?",
                )
                if not resp:
                    return
            else:
                resp = messagebox.askyesnocancel(
                    "Archivos ya procesados",
                    f"{len(ya_procesados)} de los {len(nombres_validos)} archivos seleccionados "
                    "ya tienen datos procesados.\n\n"
                    "Sí = reprocesar todos los seleccionados (incluidos esos)\n"
                    "No = procesar solo los que faltan, dejando los demás como están\n"
                    "Cancelar = no hacer nada"
                )
                if resp is None:
                    return
                if resp is False:
                    cola = [n for n in nombres_validos if n not in ya_procesados]
                    if not cola:
                        messagebox.showinfo(
                            "Nada que procesar",
                            "Todos los archivos seleccionados ya estaban procesados."
                        )
                        return

        self._liberar_cache_visual()
        self.limpiar_log()
        self.log("====================================")
        if len(cola) == 1:
            self.log(f"Procesando: {cola[0]}")
        else:
            self.log(f"Procesando {len(cola)} archivos en lote:")
            for n in cola:
                self.log(f"  - {n}")

        self._set_barra(f"Procesando 1 de {len(cola)}: {cola[0]} ...")
        self._iniciar_trabajo("Preparando procesamiento...")

        th = threading.Thread(target=self._hilo_procesar_lote, args=(cola,), daemon=True)
        th.start()

    def _hilo_procesar_lote(self, nombres_archivo):
        """
        Procesa varios archivos .dat en secuencia dentro de un solo hilo
        (uno por uno; el pipeline internamente ya usa varios núcleos para
        acelerar cada archivo individual — ver optimizaciones de rendimiento).
        Al terminar, la interfaz queda mostrando el último archivo procesado
        con éxito, y se muestra un resumen de todo el lote.
        """
        total = len(nombres_archivo)
        resultados = []  # (nombre, ok, error_o_None)
        ultimo_ok = None  # (nombre, carpeta_cache, ruta_pdf)

        def logger(msg):
            if self._cerrando:
                return
            try:
                self.raiz.after(0, lambda: self.log(str(msg)))
            except Exception:
                pass

        for i, nombre_archivo in enumerate(nombres_archivo):
            if self._cerrando:
                break

            def progress_cb(valor, mensaje=None, _i=i, _nombre=nombre_archivo):
                if self._cerrando:
                    return
                global_pct = ((_i + (float(valor) / 100.0)) / total) * 100.0
                texto = f"[{_i + 1}/{total}] {_nombre}"
                if mensaje:
                    texto += f" — {mensaje}"
                try:
                    self.raiz.after(0, lambda: self.set_progreso(global_pct, texto))
                except Exception:
                    pass

            try:
                self.raiz.after(
                    0,
                    lambda n=nombre_archivo, i=i: self._set_barra(f"Procesando {i + 1} de {total}: {n} ...")
                )
                if i > 0:
                    self.raiz.after(0, lambda n=nombre_archivo: self.log(f"------------------------------------\nProcesando: {n}"))

                carpeta_cache = procesar_archivo(
                    nombre_archivo,
                    carpeta_base=self.carpeta_dcl,
                    logger=logger,
                    progress_callback=progress_cb
                )
                if not carpeta_cache:
                    raise RuntimeError("El pipeline no devolvió la ruta de cache.")

                self.raiz.after(0, lambda: self.log("[OK] Datos procesados y guardados en cache."))
                self.raiz.after(0, lambda: self.log("[Proceso] Generando informe PDF..."))
                ruta_pdf = generar_informe_desde_cache(carpeta_cache, logger=logger)

                ultimo_ok = (nombre_archivo, carpeta_cache, ruta_pdf)
                resultados.append((nombre_archivo, True, None))

            except Exception as e:
                resultados.append((nombre_archivo, False, e))
                self.raiz.after(
                    0,
                    lambda n=nombre_archivo, err=e: self.log(f"[Error] No se pudo procesar '{n}': {err}")
                )

        def terminar_lote():
            if self._cerrando:
                return

            exitosos = [n for n, ok, _ in resultados if ok]
            fallidos = [(n, err) for n, ok, err in resultados if not ok]

            # Deja la interfaz mostrando el último archivo procesado con éxito
            if ultimo_ok:
                nombre_archivo, carpeta_cache, ruta_pdf = ultimo_ok
                self.cache_ultimo = carpeta_cache
                self.nombre_ultimo = nombre_archivo

                if ruta_pdf and os.path.exists(ruta_pdf):
                    self.ruta_pdf_ultimo = os.path.abspath(ruta_pdf)
                    self.btn_pdf.config(state="normal")
                else:
                    self.ruta_pdf_ultimo = self._buscar_pdf_preferido(nombre_archivo, carpeta_cache)
                    self.btn_pdf.config(
                        state="normal" if self.ruta_pdf_ultimo and os.path.exists(self.ruta_pdf_ultimo) else "disabled"
                    )

                self._liberar_cache_visual()
                self._er_filtrado = None
                self._limpiar_estado_welch()
                self._reset_panel_analisis()
                self._actualizar_canales_alerta()
                self._actualizar_bloques_resumen_calidad()
                self._activar_grafica_inicial()
                self.raiz.after_idle(self._actualizar_colores_lista)

            self.finalizar_progreso(
                f"Lote terminado: {len(exitosos)}/{total} procesado(s) con éxito.",
                valor=100.0,
                delay_ms=1800
            )

            if fallidos:
                detalle = "\n".join(f"• {n}: {err}" for n, err in fallidos[:10])
                if len(fallidos) > 10:
                    detalle += f"\n... y {len(fallidos) - 10} más."
                self._set_barra(f"Lote terminado con errores: {len(exitosos)} OK, {len(fallidos)} con error.")
                self.log(f"[Aviso] Lote terminado: {len(exitosos)} OK, {len(fallidos)} con error.")
                messagebox.showwarning(
                    "Procesamiento finalizado con errores",
                    f"Se procesaron {len(exitosos)} de {total} archivo(s) correctamente.\n\n"
                    f"Con errores ({len(fallidos)}):\n{detalle}\n\n"
                    f"Usa 'Vista previa' para ver el último informe generado con éxito."
                )
            else:
                self._set_barra(f"Lote terminado: {len(exitosos)}/{total} procesado(s) con éxito.")
                self.log(f"[OK] Lote terminado: {len(exitosos)}/{total} procesado(s) con éxito.")
                if total == 1:
                    mensaje = "El archivo se procesó correctamente y el informe está listo."
                else:
                    mensaje = f"Se procesaron correctamente los {len(exitosos)} archivos seleccionados."
                messagebox.showinfo(
                    "Procesado con éxito",
                    f"{mensaje}\n\nUsa 'Vista previa' para ver el último informe generado."
                )

        try:
            self.raiz.after(0, terminar_lote)
        except Exception:
            pass

    def ir_a_analisis(self):
        self._activar_tab_global("analisis")

    def ir_a_visor(self):
        self._activar_tab_global("grafica")

    def volver_y_graficar(self):
        # 1) volver a la pestaña Visor
        self.ir_a_visor()
    
        # 2) si elegiste un canal en análisis, grafícalo
        if getattr(self, "canal_analisis_simple", None):
            canal_full = self.canal_analisis_simple
            self.var_canal_nombre.set(canal_full)
            self.combo_canal.set(canal_full)
    
            # aquí sí marcamos que ya hay un "gráfico activo"
            self.graficar()

    ## =========================================================
    ## Análisis: topomaps y energía relativa
    ## =========================================================
    def _seleccionar_tab_energia_relativa(self):
        if not hasattr(self, "tabs_analisis"):
            return

        try:
            self.tabs_analisis.select(self.tab_potencia_welch)
            self._mostrar_panel_izq_analisis_actual()
        except Exception:
            pass

    def _tab_analisis_actual(self):
        if not hasattr(self, "tabs_analisis"):
            return "welch"

        try:
            tab_id = self.tabs_analisis.select()
            texto = self.tabs_analisis.tab(tab_id, "text")
        except Exception:
            return "welch"

        if texto == "Calidad técnica":
            return "calidad"
        if texto == "Proporción de bandas":
            return "ratios"
        if texto == "Potencia relativa":
            return "welch"
        return "welch"

    def _on_tabbar_grafica_click(self, idx):
        """Callback de la tabbar de Gráficas: sincroniza notebook y modo."""
        labels = list(self.modos_grafica_tabs.keys())
        texto_tab = labels[idx] if idx < len(labels) else labels[0]
        frame_tab = self.frames_tabs_grafica.get(texto_tab)
        if frame_tab is not None:
            self._sincronizando_tab_grafica = True
            try:
                self.tabs_grafica.select(frame_tab)
            finally:
                self._sincronizando_tab_grafica = False
        modo = self.modos_grafica_tabs.get(texto_tab, "Filtrado multicanal")
        self.var_modo.set(modo)
        self._on_tab_grafica()

    def _on_tabbar_analisis_click(self, idx):
        """Callback de la tabbar de Análisis: sincroniza notebook."""
        tabs = [self.tab_potencia_welch, self.tab_ratios_bandas, self.tab_calidad_senal]
        if idx < len(tabs):
            self.tabs_analisis.select(tabs[idx])
            # El notebook dispara <<NotebookTabChanged>> → _on_tab_analisis

    def _on_tab_analisis(self, _evt=None):
        tab = self._tab_analisis_actual()
        self._actualizar_barra_modo_analisis(tab)
        self._mostrar_panel_izq_analisis_actual()
        self._actualizar_visibilidad_etiquetas_analisis()
        if tab == "energia":
            self._programar_redibujo_topomap()
        elif tab == "welch":
            self.actualizar_panel_izquierdo_potencia_relativa()
            self._programar_redibujo_welch()
        elif tab == "ratios":
            self.actualizar_panel_izquierdo_ratios()
            self._programar_redibujo_ratios()
        elif tab == "calidad":
            self.mostrar_dashboard_calidad_senal(seleccionar_tab=False)

    def mostrar_analisis_potencia_relativa(self):
        """
        Activa la vista de potencia relativa Welch sin recalcular datos.
        """
        if not hasattr(self, "tabs_analisis"):
            return

        try:
            self.panel_grafica.pack_forget()
            self.panel_analisis.pack(fill=tk.BOTH, expand=True)
            self._mostrar_marco_visor_grafica(False)
            self._mostrar_tabs_grafica(False)
        except Exception:
            pass

        self._mostrar_panel_izquierdo("analisis")
        try:
            self.tabs_analisis.select(self.tab_potencia_welch)
        except Exception:
            pass
        self._mostrar_panel_izq_analisis_actual()
        self.actualizar_panel_izquierdo_potencia_relativa()
        self._programar_redibujo_welch()

    def mostrar_analisis_ratios_bandas(self):
        """
        Activa la vista de proporción de bandas usando cache Welch.
        """
        if not hasattr(self, "tabs_analisis"):
            return

        try:
            self.panel_grafica.pack_forget()
            self.panel_analisis.pack(fill=tk.BOTH, expand=True)
            self._mostrar_marco_visor_grafica(False)
            self._mostrar_tabs_grafica(False)
        except Exception:
            pass

        self._mostrar_panel_izquierdo("analisis")
        try:
            self.tabs_analisis.select(self.tab_ratios_bandas)
        except Exception:
            pass
        self._mostrar_panel_izq_analisis_actual()
        self.actualizar_panel_izquierdo_ratios()
        self._programar_redibujo_ratios()

    def _mostrar_panel_izq_analisis_actual(self):
        if not hasattr(self, "frm_energia_relativa_izq"):
            return

        tab = self._tab_analisis_actual()
        self._actualizar_visibilidad_alertas_analisis()
        self._actualizar_visibilidad_etiquetas_analisis()
        if hasattr(self, "frm_resumen_calidad_analisis"):
            if tab == "calidad":
                self.frm_resumen_calidad_analisis.pack_forget()
            else:
                self._mostrar_resumen_calidad_compacto_analisis()
        if tab == "welch":
            self.mostrar_panel_contextual_potencia_relativa()
        elif tab == "ratios":
            self.mostrar_panel_contextual_ratios_bandas()
        elif tab == "calidad":
            self.mostrar_panel_contextual_calidad()
        else:
            self.mostrar_panel_contextual_potencia_relativa()
        self._actualizar_scrollregion_izquierda()

    def _crear_tab_analisis(self):
        """
        Panel derecho de ANÁLISIS:
        - Solo visor de topomaps (Delta, Theta, Alfa, Beta)
        - Los controles y métricas van en self.panel_izq_analisis
        """
        cont = self.tab_analisis
    
        cont.columnconfigure(0, weight=1)
        cont.rowconfigure(0, weight=1)
    
        # contenedor único del visor derecho
        frm_topo_host = ttk.Frame(cont, padding=10)
        frm_topo_host.grid(row=0, column=0, sticky="nsew")
        frm_topo_host.columnconfigure(0, weight=1)
        frm_topo_host.rowconfigure(0, weight=1)
    
        self.frm_topo = ttk.Frame(frm_topo_host)
        self.frm_topo.grid(row=0, column=0, sticky="nsew")
        self.frm_topo.columnconfigure(0, weight=1)
        self.frm_topo.rowconfigure(0, weight=1)
    
        self.fig_topo = plt.Figure(figsize=(9.0, 6.8), dpi=self.dpi_pantalla)
    
        self.ax_topo_delta = None
        self.ax_topo_theta = None
        self.ax_topo_alfa = None
        self.ax_topo_beta = None
        self._ax_to_band = {}
    
        self.canvas_topo = FigureCanvasTkAgg(self.fig_topo, master=self.frm_topo)
        self.widget_topo = self.canvas_topo.get_tk_widget()
        self.widget_topo.grid(row=0, column=0, sticky="nsew")
        self.widget_topo.bind("<Configure>", self._on_topo_configure, add="+")
    
        self.canvas_topo.mpl_connect("motion_notify_event", self._on_motion_topomap)
        self.canvas_topo.mpl_connect("button_press_event", self._on_click_topomap)    

    def _crear_tab_potencia_welch(self):
        cont = self.tab_potencia_welch
        cont.columnconfigure(0, weight=1)
        cont.rowconfigure(0, weight=1)
        self.lbl_resumen_welch = None

        self.frm_welch = ttk.Frame(cont, padding=10)
        self.frm_welch.grid(row=0, column=0, sticky="nsew")
        self.frm_welch.columnconfigure(0, weight=1)
        self.frm_welch.rowconfigure(0, weight=1)

        self.fig_welch = plt.Figure(figsize=(9.0, 6.8), dpi=self.dpi_pantalla)
        self.canvas_welch = FigureCanvasTkAgg(self.fig_welch, master=self.frm_welch)
        self.widget_welch = self.canvas_welch.get_tk_widget()
        self.widget_welch.grid(row=0, column=0, sticky="nsew")
        self.widget_welch.bind("<Configure>", self._on_welch_configure, add="+")
        self.canvas_welch.mpl_connect("motion_notify_event", self._on_motion_topomap_welch)
        self.canvas_welch.mpl_connect("button_press_event", self._on_click_topomap_welch)

    def _crear_tab_ratios_bandas(self):
        cont = self.tab_ratios_bandas
        cont.columnconfigure(0, weight=1)
        cont.rowconfigure(0, weight=1)

        self.frm_ratios = ttk.Frame(cont, padding=10)
        self.frm_ratios.grid(row=0, column=0, sticky="nsew")
        self.frm_ratios.columnconfigure(0, weight=1)
        self.frm_ratios.rowconfigure(0, weight=1)

        self.fig_ratios = plt.Figure(figsize=(9.0, 6.8), dpi=self.dpi_pantalla)
        self.canvas_ratios = FigureCanvasTkAgg(self.fig_ratios, master=self.frm_ratios)
        self.widget_ratios = self.canvas_ratios.get_tk_widget()
        self.widget_ratios.grid(row=0, column=0, sticky="nsew")
        self.widget_ratios.bind("<Configure>", self._on_ratios_configure, add="+")
        self.canvas_ratios.mpl_connect("motion_notify_event", self._on_motion_topomap_ratios)
        self.canvas_ratios.mpl_connect("button_press_event", self._on_click_topomap_ratios)


    def _crear_tab_calidad_senal(self):
        cont = self.tab_calidad_senal
        cont.columnconfigure(0, weight=1)
        cont.rowconfigure(0, weight=1)
        cont.rowconfigure(1, weight=0)

        self.frm_calidad = ttk.Frame(cont, padding=6, height=390)
        self.frm_calidad.grid(row=0, column=0, sticky="nsew")
        self.frm_calidad.grid_propagate(False)
        self.frm_calidad.columnconfigure(0, weight=1)
        self.frm_calidad.rowconfigure(0, weight=1)

        self.fig_calidad = plt.Figure(figsize=(7.8, 3.6), dpi=self.dpi_pantalla)
        self.canvas_calidad = FigureCanvasTkAgg(self.fig_calidad, master=self.frm_calidad)
        self.widget_calidad = self.canvas_calidad.get_tk_widget()
        self.widget_calidad.grid(row=0, column=0, sticky="nsew")

        self.frm_calidad_resumen = ttk.Frame(cont, padding=(10, 0, 10, 10))
        self.frm_calidad_resumen.grid(row=1, column=0, sticky="ew")
        self.frm_calidad_resumen.columnconfigure(0, weight=1)

        self.lbl_calidad_dashboard_titulo = ttk.Label(
            self.frm_calidad_resumen,
            text="CONTROL DE CALIDAD DE LA SEÑAL",
            font=("Segoe UI", 12, "bold"),
            anchor="w"
        )
        self.lbl_calidad_dashboard_titulo.pack(fill=tk.X, pady=(0, 4))

        self.lbl_calidad_dashboard_estado = tk.Label(
            self.frm_calidad_resumen,
            text="Estado global: --",
            anchor="w",
            justify="left",
            font=("Segoe UI", 16, "bold"),
            bg=self.raiz.cget("bg")
        )
        self.lbl_calidad_dashboard_estado.pack(fill=tk.X, pady=(0, 10))

        fila_metricas = ttk.Frame(self.frm_calidad_resumen)
        fila_metricas.pack(fill=tk.X, pady=(0, 8))
        for col in range(3):
            fila_metricas.columnconfigure(col, weight=1)

        self.card_calidad_afectados = ttk.LabelFrame(fila_metricas, text="Índice técnico ponderado", padding=8)
        self.card_calidad_afectados.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.lbl_calidad_dashboard_pct = ttk.Label(
            self.card_calidad_afectados,
            text="--",
            font=("Segoe UI", 12, "bold"),
            anchor="center"
        )
        self.lbl_calidad_dashboard_pct.pack(fill=tk.X)

        self.card_calidad_snr = ttk.LabelFrame(fila_metricas, text="SNR global", padding=8)
        self.card_calidad_snr.grid(row=0, column=1, sticky="nsew", padx=3)
        self.lbl_calidad_dashboard_snr = ttk.Label(
            self.card_calidad_snr,
            text="--",
            font=("Segoe UI", 12, "bold"),
            anchor="center"
        )
        self.lbl_calidad_dashboard_snr.pack(fill=tk.X)

        self.card_calidad_region = ttk.LabelFrame(fila_metricas, text="Región predominante", padding=8)
        self.card_calidad_region.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.lbl_calidad_dashboard_region = ttk.Label(
            self.card_calidad_region,
            text="--",
            font=("Segoe UI", 11, "bold"),
            anchor="center"
        )
        self.lbl_calidad_dashboard_region.pack(fill=tk.X)

        self.lbl_calidad_dashboard_componentes = ttk.Label(
            self.frm_calidad_resumen,
            text="Atípicos: -- | Sospechosos: --",
            anchor="w",
            justify="left",
            style="Sidebar.TLabel"
        )
        self.lbl_calidad_dashboard_componentes.pack(fill=tk.X, pady=(0, 8))

        self.card_calidad_snr_interpretacion = ttk.LabelFrame(
            self.frm_calidad_resumen,
            text="Interpretación de SNR",
            padding=10
        )
        self.card_calidad_snr_interpretacion.pack(fill=tk.X)
        self.lbl_calidad_dashboard_snr_texto = ttk.Label(
            self.card_calidad_snr_interpretacion,
            text="--",
            anchor="w",
            justify="left",
            wraplength=1100
        )
        self.lbl_calidad_dashboard_snr_texto.pack(fill=tk.X)
 
    def _crear_panel_izq_analisis(self):
        if getattr(self, "_panel_izq_analisis_creado", False):
            return
        self._panel_izq_analisis_creado = True
        
        self.frm_alertas_analisis = ttk.Frame(self.panel_izq_analisis)

        self.lbl_atipicos_analisis = ttk.Label(
            self.frm_alertas_analisis,
            textvariable=self.var_atipicos,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_atipicos_analisis.pack(fill=tk.X, pady=(0, 4))
        
        self.lbl_sospechosos_analisis = ttk.Label(
            self.frm_alertas_analisis,
            textvariable=self.var_sospechosos,
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_sospechosos_analisis.pack(fill=tk.X)
        self.frm_alertas_analisis.pack_forget()
        self.frm_resumen_calidad_analisis = self._crear_bloque_resumen_calidad(
            self.panel_izq_analisis,
            "analisis"
        )
        self.frm_resumen_calidad_analisis.pack(fill=tk.X, pady=(0, 8))
        ## Sección de opciones de topomapas
        self._frm_opciones_topo = ttk.LabelFrame(
            self.panel_izq_analisis,
            text="Opciones de topomapa",
            padding=(10, 6, 10, 8),
            style="Card.TLabelframe",
        )
        self._frm_opciones_topo.pack(fill=tk.X, pady=(0, 8))
        self.chk_etiquetas_topo = ttk.Checkbutton(
            self._frm_opciones_topo,
            text="Mostrar etiquetas de canal",
            variable=self.var_etiquetas_topo,
            command=self._toggle_etiquetas_topo,
        )
        self.chk_etiquetas_topo.pack(anchor="w")
    
        self.frm_energia_relativa_izq = ttk.LabelFrame(self.panel_izq_analisis, text="Energía relativa", padding=10)
        self.frm_energia_relativa_izq.pack(fill=tk.X)
    
        self.lbl_sel_canal = ttk.Label(
            self.frm_energia_relativa_izq,
            text="Canal: (haz clic en el mapa)",
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_sel_canal.pack(anchor="w", pady=(0, 8))
    
        self.lbl_bandas = {}
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            lab = ttk.Label(self.frm_energia_relativa_izq, text=f"{b}: --%")
            lab.pack(anchor="w")
            self.lbl_bandas[b] = lab
    
        ttk.Separator(self.frm_energia_relativa_izq).pack(fill=tk.X, pady=8)
    
        self.lbl_region = ttk.Label(self.frm_energia_relativa_izq, text="Región: --", font=("Segoe UI", 9, "bold"))
        self.lbl_region.pack(anchor="w", pady=(0, 6))
    
        self.lbl_region_bandas = {}
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            lab = ttk.Label(self.frm_energia_relativa_izq, text=f"{b} (región): --%")
            lab.pack(anchor="w")
            self.lbl_region_bandas[b] = lab

        self._crear_panel_izq_potencia_welch()
        self._crear_panel_izq_ratios_bandas()
        self._crear_panel_izq_calidad()

    def _crear_panel_izq_potencia_welch(self):
        self.frm_potencia_welch_izq = ttk.LabelFrame(
            self.panel_izq_analisis,
            text="Potencia relativa",
            padding=10
        )
        self.frm_potencia_welch_izq.configure(style="Card.TLabelframe", text="Potencia relativa")

        self.lbl_subtitulo_welch_izq = ttk.Label(
            self.frm_potencia_welch_izq,
            text="Valores calculados mediante Welch",
            style="Sidebar.TLabel",
            anchor="w",
            justify="left"
        )
        self.lbl_subtitulo_welch_izq.pack(fill=tk.X, pady=(0, 4))

        self.lbl_estado_welch_izq = ttk.Label(
            self.frm_potencia_welch_izq,
            text="Sin datos cargados.",
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_estado_welch_izq.pack(fill=tk.X, pady=(0, 8))

        self.frm_welch_global = ttk.Frame(self.frm_potencia_welch_izq, style="Card.TFrame", padding=8)
        self.frm_welch_global.pack(fill=tk.X, pady=(0, 8))

        self.lbl_welch_global_titulo = ttk.Label(
            self.frm_welch_global,
            text="Resumen global",
            font=(self.fuente_ui, 9, "bold")
        )
        self.lbl_welch_global_titulo.pack(anchor="w", pady=(0, 4))

        self.tabla_welch_global = ttk.Treeview(
            self.frm_welch_global,
            columns=("banda", "valor"),
            show="headings",
            height=4,
            selectmode="none",
            style="NeuroX.Treeview"
        )
        self.tabla_welch_global.heading("banda", text="Banda")
        self.tabla_welch_global.heading("valor", text="Valor global")
        self.tabla_welch_global.column("banda", width=104, minwidth=88, stretch=True, anchor="w")
        self.tabla_welch_global.column("valor", width=112, minwidth=96, stretch=True, anchor="center")
        self.tabla_welch_global.pack(fill=tk.X)
        self._aplicar_estilo_treeview(self.tabla_welch_global)

        self.frm_welch_regional = ttk.Frame(self.frm_potencia_welch_izq, style="Card.TFrame", padding=8)
        self.frm_welch_regional.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.frm_welch_regional,
            text="Promedios regionales",
            font=(self.fuente_ui, 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        self.regiones_welch = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
        columnas = ("region", "delta", "theta", "alfa", "beta")
        self.tabla_welch_regiones = ttk.Treeview(
            self.frm_welch_regional,
            columns=columnas,
            show="headings",
            height=len(self.regiones_welch),
            selectmode="none",
            style="NeuroX.Treeview"
        )
        encabezados = {
            "region": "Región",
            "delta": "Delta",
            "theta": "Theta",
            "alfa": "Alfa",
            "beta": "Beta",
        }
        anchos = {
            "region": 76,
            "delta": 52,
            "theta": 52,
            "alfa": 52,
            "beta": 52,
        }
        min_anchos = {
            "region": 70,
            "delta": 50,
            "theta": 50,
            "alfa": 50,
            "beta": 50,
        }
        for col in columnas:
            self.tabla_welch_regiones.heading(col, text=encabezados[col])
            self.tabla_welch_regiones.column(
                col,
                width=anchos[col],
                minwidth=min_anchos[col],
                stretch=True,
                anchor="center"
            )
        self.tabla_welch_regiones.column("region", anchor="w")
        self.tabla_welch_regiones.pack(fill=tk.X, anchor="w")
        self._aplicar_estilo_treeview(self.tabla_welch_regiones)
        self._actualizar_tabla_regiones_welch()

        self.frm_welch_canal = ttk.Frame(self.frm_potencia_welch_izq, style="Card.TFrame", padding=8)
        self.frm_welch_canal.pack(fill=tk.X)

        self.lbl_welch_sel_canal = ttk.Label(
            self.frm_welch_canal,
            text="Canal seleccionado: --",
            font=(self.fuente_ui, 10, "bold")
        )
        self.lbl_welch_sel_canal.pack(anchor="w", pady=(0, 8))

        self.lbl_welch_bandas = {}
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            lab = ttk.Label(self.frm_welch_canal, text=f"{b}: --%")
            lab.pack(anchor="w")
            self.lbl_welch_bandas[b] = lab

    def _crear_panel_izq_ratios_bandas(self):
        self.frm_ratios_bandas_izq = ttk.LabelFrame(
            self.panel_izq_analisis,
            text="Proporción de bandas",
            padding=10
        )
        self.frm_ratios_bandas_izq.configure(style="Card.TLabelframe", text="Proporción de bandas")

        self.lbl_subtitulo_ratios_izq = ttk.Label(
            self.frm_ratios_bandas_izq,
            text="Relaciones derivadas desde la potencia relativa en cache Welch",
            style="Sidebar.TLabel",
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_subtitulo_ratios_izq.pack(fill=tk.X, pady=(0, 4))

        self.lbl_estado_ratios_izq = ttk.Label(
            self.frm_ratios_bandas_izq,
            text="Sin datos cargados.",
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_estado_ratios_izq.pack(fill=tk.X, pady=(0, 8))

        self.frm_ratios_global = ttk.Frame(self.frm_ratios_bandas_izq, style="Card.TFrame", padding=8)
        self.frm_ratios_global.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.frm_ratios_global,
            text="Proporciones globales",
            font=(self.fuente_ui, 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        columnas = ("ratio", "valor")
        self.tabla_ratios_global = ttk.Treeview(
            self.frm_ratios_global,
            columns=columnas,
            show="headings",
            height=4,
            selectmode="none",
            style="NeuroX.Treeview"
        )
        self.tabla_ratios_global.heading("ratio", text="Ratio")
        self.tabla_ratios_global.heading("valor", text="Global")
        self.tabla_ratios_global.column("ratio", width=138, minwidth=118, stretch=True, anchor="w")
        self.tabla_ratios_global.column("valor", width=78, minwidth=64, stretch=True, anchor="center")
        self.tabla_ratios_global.pack(fill=tk.X)
        self._aplicar_estilo_treeview(self.tabla_ratios_global)

        self.frm_ratios_regional = ttk.Frame(self.frm_ratios_bandas_izq, style="Card.TFrame", padding=8)
        self.frm_ratios_regional.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.frm_ratios_regional,
            text="Promedios regionales",
            font=(self.fuente_ui, 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        columnas_reg = ("region", "theta_alfa", "delta_alfa", "theta_beta", "lent")
        self.tabla_ratios_regiones = ttk.Treeview(
            self.frm_ratios_regional,
            columns=columnas_reg,
            show="headings",
            height=5,
            selectmode="none",
            style="NeuroX.Treeview"
        )
        encabezados = {
            "region": "Región",
            "theta_alfa": "T/A",
            "delta_alfa": "D/A",
            "theta_beta": "T/B",
            "lent": "Lent.",
        }
        anchos = {
            "region": 78,
            "theta_alfa": 48,
            "delta_alfa": 48,
            "theta_beta": 48,
            "lent": 52,
        }
        for col in columnas_reg:
            self.tabla_ratios_regiones.heading(col, text=encabezados[col])
            self.tabla_ratios_regiones.column(
                col,
                width=anchos[col],
                minwidth=max(44, anchos[col] - 8),
                stretch=True,
                anchor="center"
            )
        self.tabla_ratios_regiones.column("region", anchor="w")
        self.tabla_ratios_regiones.pack(fill=tk.X)
        self._aplicar_estilo_treeview(self.tabla_ratios_regiones)

        self.frm_ratios_personalizado = ttk.Frame(self.frm_ratios_bandas_izq, style="Card.TFrame", padding=8)
        self.frm_ratios_personalizado.pack(fill=tk.X, pady=(0, 8))

        self.btn_personalizar_ratio = ttk.Button(
            self.frm_ratios_personalizado,
            text="Personalizar proporción",
            command=self._activar_personalizacion_ratio
        )
        self.btn_personalizar_ratio.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.frm_ratios_personalizado,
            text="Proporción personalizada",
            font=(self.fuente_ui, 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        opciones = ["Delta", "Theta", "Alfa", "Beta", "Delta+Theta", "Alfa+Beta"]
        self.var_ratio_num = tk.StringVar(value="Theta")
        self.var_ratio_den = tk.StringVar(value="Alfa")

        fila_num = ttk.Frame(self.frm_ratios_personalizado, style="Card.TFrame")
        fila_num.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(fila_num, text="Numerador:", width=12).pack(side=tk.LEFT)
        self.combo_ratio_num = ttk.Combobox(
            fila_num,
            textvariable=self.var_ratio_num,
            values=opciones,
            state="readonly",
            width=16
        )
        self.combo_ratio_num.pack(side=tk.LEFT, fill=tk.X, expand=True)

        fila_den = ttk.Frame(self.frm_ratios_personalizado, style="Card.TFrame")
        fila_den.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(fila_den, text="Denominador:", width=12).pack(side=tk.LEFT)
        self.combo_ratio_den = ttk.Combobox(
            fila_den,
            textvariable=self.var_ratio_den,
            values=opciones,
            state="readonly",
            width=16
        )
        self.combo_ratio_den.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.combo_ratio_num.bind("<<ComboboxSelected>>", self._on_ratio_personalizado_change)
        self.combo_ratio_den.bind("<<ComboboxSelected>>", self._on_ratio_personalizado_change)

        self.lbl_ratio_activa = ttk.Label(
            self.frm_ratios_personalizado,
            text="Proporción activa: Theta / Alfa",
            anchor="w",
            justify="left",
            wraplength=280,
            font=(self.fuente_ui, 9, "bold")
        )
        self.lbl_ratio_activa.pack(fill=tk.X, pady=(0, 8))

        self.btn_ver_ratios_principales = ttk.Button(
            self.frm_ratios_personalizado,
            text="Ver 4 proporciones principales",
            command=self._mostrar_ratios_principales
        )
        self.btn_ver_ratios_principales.pack(fill=tk.X, pady=(0, 8))

        self.frm_ratios_canal = ttk.Frame(self.frm_ratios_bandas_izq, style="Card.TFrame", padding=8)
        self.frm_ratios_canal.pack(fill=tk.X)

        self.lbl_ratio_sel_canal = ttk.Label(
            self.frm_ratios_canal,
            text="Canal seleccionado: --",
            font=(self.fuente_ui, 10, "bold")
        )
        self.lbl_ratio_sel_canal.pack(anchor="w", pady=(0, 4))

        self.lbl_ratio_valor_canal = ttk.Label(
            self.frm_ratios_canal,
            text="Proporción activa: --",
            anchor="w",
            justify="left",
            wraplength=280
        )

        self.lbl_ratios_canal = {}
        for key, label in self._ratios_labels().items():
            lab = ttk.Label(self.frm_ratios_canal, text=f"{label}: --")
            lab.pack(anchor="w")
            self.lbl_ratios_canal[key] = lab

    def _crear_panel_izq_calidad(self):
        self.frm_calidad_senal_izq = ttk.LabelFrame(self.panel_izq_analisis, text="Calidad técnica", padding=10)

        self.frm_calidad_senal_izq.configure(style="Card.TLabelframe", text="Calidad técnica")
        self.lbl_calidad_estado = ttk.Label(
            self.frm_calidad_senal_izq,
            text="Estado global: --",
            font=(self.fuente_ui, 10, "bold"),
            anchor="w",
            foreground=self._color_estado_calidad(None)
        )
        self.lbl_calidad_estado.pack(fill=tk.X, pady=(0, 6))

        self.lbl_calidad_pct = ttk.Label(self.frm_calidad_senal_izq, text="Índice técnico ponderado: --")
        self.lbl_calidad_pct.pack(anchor="w")

        self.lbl_calidad_pct_detalle = ttk.Label(
            self.frm_calidad_senal_izq,
            text="Atípicos: -- | Sospechosos: --",
            style="Sidebar.TLabel"
        )
        self.lbl_calidad_pct_detalle.pack(anchor="w")

        self.lbl_calidad_snr = ttk.Label(self.frm_calidad_senal_izq, text="SNR global: --")
        self.lbl_calidad_snr.pack(anchor="w")

        self.lbl_calidad_region = ttk.Label(self.frm_calidad_senal_izq, text="Región predominante: --", wraplength=280)
        self.lbl_calidad_region.pack(anchor="w", pady=(0, 8))

        ttk.Separator(self.frm_calidad_senal_izq).pack(fill=tk.X, pady=8)

        self.lbl_calidad_atipicos = ttk.Label(
            self.frm_calidad_senal_izq,
            text="Canales atípicos: --",
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_calidad_atipicos.pack(fill=tk.X)

        self.lbl_calidad_sospechosos = ttk.Label(
            self.frm_calidad_senal_izq,
            text="Canales sospechosos: --",
            anchor="w",
            justify="left",
            wraplength=280
        )
        self.lbl_calidad_sospechosos.pack(fill=tk.X, pady=(6, 0))
        self.lbl_calidad_atipicos.config(text="Canales atípicos:\n--", wraplength=296)
        self.lbl_calidad_sospechosos.config(text="Canales sospechosos:\n--", wraplength=296)
    
    def _nombre_simple(self, nombre_canal: str) -> str:
        """Convierte variantes visibles como 'FP1-avg' a 'FP1'."""
        return limpiar_nombre_canal(nombre_canal)

    def _formatear_pct(self, valor):
        try:
            return f"{float(valor):.1f}".replace(".", ",") + "%"
        except Exception:
            return "--%"

    def _formatear_ratio(self, valor):
        try:
            return f"{float(valor):.1f}".replace(".", ",")
        except Exception:
            return "--"

    def _ratios_labels(self):
        return {
            "Theta_Alfa": "Theta/Alfa",
            "Delta_Alfa": "Delta/Alfa",
            "Theta_Beta": "Theta/Beta",
            "Lentificacion": "Lentificación",
        }

    def _nombre_ratio_personalizado(self):
        num = self.var_ratio_num.get() if hasattr(self, "var_ratio_num") else "Theta"
        den = self.var_ratio_den.get() if hasattr(self, "var_ratio_den") else "Alfa"
        return f"{num} / {den}"

    def _reset_panel_izquierdo_ratios(self, mensaje="Sin datos cargados."):
        if not hasattr(self, "lbl_estado_ratios_izq"):
            return

        self.lbl_estado_ratios_izq.config(text=mensaje)
        self._actualizar_tabla_ratios_global()
        self._actualizar_tabla_ratios_regiones()
        self.lbl_ratio_activa.config(text=f"Proporción activa: {self._nombre_ratio_personalizado()}")
        self.lbl_ratio_sel_canal.config(text="Canal seleccionado: --")
        self.lbl_ratio_valor_canal.config(text="Proporción activa: --")
        self._actualizar_visibilidad_ratio_personalizado_canal()
        for key, lab in self.lbl_ratios_canal.items():
            lab.config(text=f"{self._ratios_labels().get(key, key)}: --")

    def _actualizar_tabla_ratios_global(self, promedios=None):
        if not hasattr(self, "tabla_ratios_global"):
            return

        for item in self.tabla_ratios_global.get_children():
            self.tabla_ratios_global.delete(item)

        if promedios is None:
            promedios = {}

        for key, label in self._ratios_labels().items():
            self._insertar_fila_treeview(
                self.tabla_ratios_global,
                (label, self._formatear_ratio(promedios.get(key, None)))
            )

    def _actualizar_tabla_ratios_regiones(self, promedios_regionales=None):
        if not hasattr(self, "tabla_ratios_regiones"):
            return

        for item in self.tabla_ratios_regiones.get_children():
            self.tabla_ratios_regiones.delete(item)

        if promedios_regionales is None:
            promedios_regionales = {}

        for reg in ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]:
            vals = promedios_regionales.get(reg, {})
            fila = [
                reg,
                self._formatear_ratio(vals.get("Theta_Alfa", None)),
                self._formatear_ratio(vals.get("Delta_Alfa", None)),
                self._formatear_ratio(vals.get("Theta_Beta", None)),
                self._formatear_ratio(vals.get("Lentificacion", None)),
            ]
            self._insertar_fila_treeview(self.tabla_ratios_regiones, fila)

    def _sumar_componentes_ratio(self, componente):
        if self._pot_rel_welch is None:
            return None

        partes = componente.split("+")
        total = None
        for parte in partes:
            parte = parte.strip()
            if parte not in self._pot_rel_welch:
                return None
            arr = np.asarray(self._pot_rel_welch[parte], dtype=np.float64)
            total = arr.copy() if total is None else total + arr
        return total

    def calcular_ratio_personalizado_desde_pot_rel(self, numerador=None, denominador=None, eps=1e-12):
        """
        Construye una proporción al vuelo desde pot_rel_welch cargado en cache.
        No recalcula Welch ni modifica el cache.
        """
        if self._pot_rel_welch is None:
            ok = self._cargar_potencia_relativa_welch_cache()
            if not ok:
                return None

        numerador = numerador or self.var_ratio_num.get()
        denominador = denominador or self.var_ratio_den.get()

        num = self._sumar_componentes_ratio(numerador)
        den = self._sumar_componentes_ratio(denominador)
        if num is None or den is None:
            return None

        den_safe = np.where(np.abs(den) <= eps, eps, den)
        return (num / den_safe).astype(np.float64)

    def _calcular_ratios_principales_desde_pot_rel(self):
        if self._pot_rel_welch is None:
            ok = self._cargar_potencia_relativa_welch_cache()
            if not ok:
                return None

        specs = {
            "Theta_Alfa": ("Theta", "Alfa"),
            "Delta_Alfa": ("Delta", "Alfa"),
            "Theta_Beta": ("Theta", "Beta"),
            "Lentificacion": ("Delta+Theta", "Alfa+Beta"),
        }
        ratios = {}
        for key, (num, den) in specs.items():
            ratios[key] = self.calcular_ratio_personalizado_desde_pot_rel(num, den)
        return ratios

    def cargar_ratios_bandas_principales_desde_cache(self):
        """
        Carga las 4 proporciones principales desde cache.
        Si el cache antiguo no las trae, las construye desde pot_rel_welch.
        """
        self._ratios_principales = None
        self._resumen_ratios = None

        if not self.cache_ultimo:
            datos = self._calcular_ratios_principales_desde_pot_rel()
            if not datos:
                return False
            self._ratios_principales = datos
            self._resumen_ratios = None
            return True

        carpeta = os.path.join(self.cache_ultimo, "analisis_welch")
        ruta_npz = os.path.join(carpeta, "ratios_welch_principales.npz")
        ruta_json = os.path.join(carpeta, "ratios_promedios_principales.json")

        try:
            datos = {}
            if os.path.exists(ruta_npz):
                with np.load(ruta_npz, allow_pickle=False) as z:
                    for key in self._ratios_labels().keys():
                        if key not in z:
                            datos = {}
                            break
                        datos[key] = np.asarray(z[key], dtype=np.float64)

            if not datos:
                datos = self._calcular_ratios_principales_desde_pot_rel()
                if not datos:
                    return False

            resumen = None
            if os.path.exists(ruta_json):
                with open(ruta_json, "r", encoding="utf-8") as f:
                    resumen = json.load(f)

            self._ratios_principales = datos
            self._resumen_ratios = resumen
            return True
        except Exception as e:
            self.log(f"No pude cargar proporciones de bandas: {e}")
            self._ratios_principales = None
            self._resumen_ratios = None
            return False

    def _promedios_globales_ratios(self):
        if self._ratios_principales is None:
            return {}

        if isinstance(self._resumen_ratios, dict):
            proms = self._resumen_ratios.get("promedios_globales", {})
            if proms:
                return proms

        return {
            key: float(np.nanmean(np.asarray(valores, dtype=np.float64)))
            for key, valores in self._ratios_principales.items()
        }

    def _promedios_regionales_ratios(self):
        regiones_objetivo = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]

        if self._ratios_principales is None:
            return {}

        if isinstance(self._resumen_ratios, dict):
            regiones_cache = self._resumen_ratios.get("promedios_regionales", {})
            if regiones_cache:
                return {reg: regiones_cache.get(reg, {}) for reg in regiones_objetivo}

        grupos = {reg: [] for reg in regiones_objetivo}
        for i, nombre in enumerate(self.nombres_canales):
            reg = self._region_de_canal_ui(self._nombre_simple(nombre))
            if reg in grupos:
                grupos[reg].append(i)

        proms = {}
        for reg, idxs in grupos.items():
            proms[reg] = {}
            for key, valores in self._ratios_principales.items():
                arr = np.asarray(valores, dtype=np.float64)
                if len(idxs) and arr.size > max(idxs):
                    proms[reg][key] = float(np.nanmean(arr[idxs]))
                else:
                    proms[reg][key] = float("nan")
        return proms

    def actualizar_panel_izquierdo_ratios(self, canal_simple=None):
        """
        Actualiza la barra izquierda contextual de Proporción de bandas.
        """
        if not hasattr(self, "lbl_estado_ratios_izq"):
            return

        if self._ratios_principales is None:
            ok = self.cargar_ratios_bandas_principales_desde_cache()
            if not ok:
                self._reset_panel_izquierdo_ratios(
                    "No se encontró análisis de proporción de bandas para este archivo."
                )
                return

        self._ratio_personalizado = self.calcular_ratio_personalizado_desde_pot_rel()
        if self._ratio_personalizado is None:
            self._reset_panel_izquierdo_ratios(
                "No se pudo construir la proporción personalizada desde pot_rel_welch."
            )
            return

        self.lbl_estado_ratios_izq.config(text="Datos cargados desde cache Welch.")
        self.lbl_ratio_activa.config(text=f"Proporción activa: {self._nombre_ratio_personalizado()}")
        self._actualizar_tabla_ratios_global(self._promedios_globales_ratios())
        self._actualizar_tabla_ratios_regiones(self._promedios_regionales_ratios())

        canal = self._nombre_simple(canal_simple) if canal_simple else ""
        if not canal:
            canal = self.canal_ratios_simple or self.canal_analisis_simple or self.canal_activo_simple
        if not canal and self.nombres_canales:
            canal = self._nombre_simple(self.nombres_canales[0])

        self.actualizar_canal_ratios(canal, refrescar_resumen=False)

    def actualizar_canal_ratios(self, canal_simple, refrescar_resumen=True):
        canal_simple = self._nombre_simple(canal_simple)
        if not canal_simple:
            return

        if self._ratios_principales is None:
            ok = self.cargar_ratios_bandas_principales_desde_cache()
            if not ok:
                self._reset_panel_izquierdo_ratios(
                    "No se encontró análisis de proporción de bandas para este archivo."
                )
                return

        if self._ratio_personalizado is None:
            self._ratio_personalizado = self.calcular_ratio_personalizado_desde_pot_rel()

        if refrescar_resumen:
            self.actualizar_panel_izquierdo_ratios(canal_simple)
            return

        idx = None
        for i, nombre in enumerate(self.nombres_canales):
            if self._nombre_simple(nombre) == canal_simple:
                idx = i
                break

        self.canal_ratios_simple = canal_simple
        self.canal_analisis_simple = canal_simple
        self.lbl_ratio_sel_canal.config(text=f"Canal seleccionado: {canal_simple}")

        if idx is None:
            self._actualizar_visibilidad_ratio_personalizado_canal()
            self.lbl_ratio_valor_canal.config(text="Proporción activa: --")
            for key, lab in self.lbl_ratios_canal.items():
                lab.config(text=f"{self._ratios_labels().get(key, key)}: --")
            return

        self._actualizar_visibilidad_ratio_personalizado_canal()
        if self._ratio_personalizado_activo:
            try:
                v_activo = float(np.asarray(self._ratio_personalizado, dtype=np.float64)[idx])
                self.lbl_ratio_valor_canal.config(
                    text=f"{self._nombre_ratio_personalizado()}: {self._formatear_ratio(v_activo)}"
                )
            except Exception:
                self.lbl_ratio_valor_canal.config(text=f"{self._nombre_ratio_personalizado()}: --")

        for key, lab in self.lbl_ratios_canal.items():
            try:
                v = float(np.asarray(self._ratios_principales[key], dtype=np.float64)[idx])
                lab.config(text=f"{self._ratios_labels().get(key, key)}: {self._formatear_ratio(v)}")
            except Exception:
                lab.config(text=f"{self._ratios_labels().get(key, key)}: --")

        try:
            self.var_canal_nombre.set(canal_simple)
            self.combo_canal.set(canal_simple)
        except Exception:
            pass

    def _actualizar_visibilidad_ratio_personalizado_canal(self):
        if not hasattr(self, "lbl_ratio_valor_canal"):
            return

        if self._ratio_personalizado_activo:
            if not self.lbl_ratio_valor_canal.winfo_ismapped():
                self.lbl_ratio_valor_canal.pack(fill=tk.X, pady=(0, 8), before=self.lbl_ratios_canal["Theta_Alfa"])
            if hasattr(self, "btn_personalizar_ratio"):
                self.btn_personalizar_ratio.config(state="disabled")
            if hasattr(self, "btn_ver_ratios_principales"):
                self.btn_ver_ratios_principales.config(state="normal")
        else:
            self.lbl_ratio_valor_canal.pack_forget()
            if hasattr(self, "btn_personalizar_ratio"):
                self.btn_personalizar_ratio.config(state="normal")
            if hasattr(self, "btn_ver_ratios_principales"):
                self.btn_ver_ratios_principales.config(state="disabled")

    def _activar_personalizacion_ratio(self):
        self._ratio_personalizado_activo = True
        self._ratio_personalizado = self.calcular_ratio_personalizado_desde_pot_rel()
        self._actualizar_visibilidad_ratio_personalizado_canal()
        self.actualizar_panel_izquierdo_ratios(self.canal_ratios_simple)
        self._programar_redibujo_ratios()

    def _on_ratio_personalizado_change(self, _evt=None):
        if not self._ratio_personalizado_activo:
            return
        self._ratio_personalizado_activo = True
        self._ratio_personalizado = self.calcular_ratio_personalizado_desde_pot_rel()
        self._actualizar_visibilidad_ratio_personalizado_canal()
        self.actualizar_panel_izquierdo_ratios(self.canal_ratios_simple)
        self._programar_redibujo_ratios()

    def _mostrar_ratios_principales(self):
        self._ratio_personalizado_activo = False
        self._ratio_personalizado = None
        self._actualizar_visibilidad_ratio_personalizado_canal()
        self.actualizar_panel_izquierdo_ratios(self.canal_ratios_simple)
        self._programar_redibujo_ratios()

    def _pares_homologos_base_ui(self):
        return [
            ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F5", "F6"),
            ("F3", "F4"), ("F1", "F2"), ("FC5", "FC6"), ("FC3", "FC4"),
            ("FC1", "FC2"), ("C5", "C6"), ("C3", "C4"), ("C1", "C2"),
            ("CP5", "CP6"), ("CP3", "CP4"), ("CP1", "CP2"), ("P7", "P8"),
            ("P5", "P6"), ("P3", "P4"), ("P1", "P2"), ("PO7", "PO8"),
            ("PO5", "PO6"), ("PO3", "PO4"), ("O1", "O2"), ("CB1", "CB2"),
        ]

    def _set_texto_calidad(self, texto):
        self._texto_calidad = texto

    def _reset_panel_izquierdo_calidad(self, mensaje="Sin datos cargados."):
        if not hasattr(self, "lbl_calidad_estado"):
            return
        self.lbl_calidad_estado.config(text="Calidad técnica: no disponible.")
        self.lbl_calidad_estado.config(foreground=self._color_estado_calidad(None))
        self.lbl_calidad_pct.config(text="Índice técnico ponderado: --")
        if hasattr(self, "lbl_calidad_pct_detalle"):
            self.lbl_calidad_pct_detalle.config(text="Atípicos: -- | Sospechosos: --")
        self.lbl_calidad_snr.config(text="SNR global: --")
        self.lbl_calidad_region.config(text="Región predominante: --")
        self.lbl_calidad_atipicos.config(text="Canales atípicos: --")
        self.lbl_calidad_sospechosos.config(text="Canales sospechosos: --")
        self.lbl_calidad_atipicos.config(text="Canales atípicos:\n--")
        self.lbl_calidad_sospechosos.config(text="Canales sospechosos:\n--")
        self._set_texto_calidad(mensaje)
        self._reset_resumen_visual_calidad(mensaje)

    def _dibujar_mapa_calidad_electrodos(self, ax, resumen):
        escala_mapa = 0.78
        radio_cabeza = escala_mapa

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.20, 1.20)
        ax.set_ylim(-1.20, 1.20)
        ax.axis("off")
        ax.set_title("Mapa 10-10 de calidad", fontsize=9)

        cabeza = plt.Circle((0, 0), radio_cabeza, fill=False, color="black", linewidth=1.6)
        ax.add_patch(cabeza)
        ax.plot(
            [-0.08 * radio_cabeza, 0.0, 0.08 * radio_cabeza],
            [1.00 * radio_cabeza, 1.10 * radio_cabeza, 1.00 * radio_cabeza],
            color="black",
            lw=1.6
        )

        t = np.linspace(-0.9, 0.9, 80)
        ax.plot(
            -radio_cabeza - 0.015 * radio_cabeza * np.cos(t),
            0.20 * radio_cabeza * np.sin(t),
            color="black",
            lw=1.8,
            clip_on=True
        )
        ax.plot(
            radio_cabeza + 0.015 * radio_cabeza * np.cos(t),
            0.20 * radio_cabeza * np.sin(t),
            color="black",
            lw=1.8,
            clip_on=True
        )

        atipicos = {self._nombre_simple(c) for c in resumen.get("canales_atipicos", [])}
        sospechosos = {self._nombre_simple(c) for c in resumen.get("canales_sospechosos", [])}

        dibujados = []
        for nombre in self.nombres_canales:
            simple = self._nombre_simple(nombre)
            p = self._coord_1010_curvada_norm(simple)
            if p is None:
                continue
            x, y = p
            x *= escala_mapa
            y *= escala_mapa

            if simple in atipicos:
                color = "#DC2626"
                size = 24
            elif simple in sospechosos:
                color = "#F59E0B"
                size = 22
            else:
                color = "#22C55E"
                size = 18

            ax.scatter([x], [y], s=size, c=color, edgecolors="black", linewidths=0.6, zorder=3)
            dibujados.append((simple, x, y))

        for simple, x, y in dibujados:
            ax.text(x, y + 0.024, simple, fontsize=5, ha="center", va="bottom")

        legend_handles = [
            Patch(facecolor="#22C55E", edgecolor="black", label="Normal"),
            Patch(facecolor="#F59E0B", edgecolor="black", label="Sospechoso"),
            Patch(facecolor="#DC2626", edgecolor="black", label="Atípico"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            fontsize=7,
            frameon=False
        )

    def cargar_resumen_calidad(self, mostrar_warning=False):
        import os
        import json

        if not self.cache_ultimo:
            if mostrar_warning:
                messagebox.showwarning("Sin archivo", "Primero procesa o selecciona un archivo.")
            return None

        ruta_json = os.path.join(self.cache_ultimo, "calidad_senal", "resumen_calidad.json")

        if not os.path.exists(ruta_json):
            if mostrar_warning:
                messagebox.showwarning(
                    "Sin calidad de señal",
                    "No se encontró el resumen de calidad. Procesa nuevamente el archivo."
                )
            return None

        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                resumen = json.load(f)
            self._resumen_calidad = resumen
            return resumen
        except Exception as e:
            self.log(f"No pude cargar resumen de calidad: {e}")
            if mostrar_warning:
                messagebox.showwarning("Calidad técnica", f"No pude leer el resumen de calidad:\n{e}")
            return None

    def _texto_dashboard_calidad(self, resumen):
        metricas = self._metricas_calidad_resumen(resumen)
        estado = metricas["estado"]
        porc_afectado = metricas["pct_ponderado"]
        snr_db = metricas["snr_db"]
        region_pred = metricas["region"]
        pct_atipicos = metricas["pct_atipicos"]
        pct_sospechosos = metricas["pct_sospechosos"]

        canales_atipicos = ", ".join(resumen.get("canales_atipicos", [])) or "Ninguno"
        canales_sospechosos = ", ".join(resumen.get("canales_sospechosos", [])) or "Ninguno"

        return (
            "CONTROL DE CALIDAD DE LA SEÑAL\n\n"
            f"Estado global: {estado}\n"
            f"Canales atípicos fuertes: {pct_atipicos:.1f}%\n"
            f"Canales sospechosos: {pct_sospechosos:.1f}%\n"
            f"Índice técnico ponderado: {porc_afectado:.1f}%\n"
            f"SNR global estimada: {snr_db:.2f} dB\n"
            f"Región predominante afectada: {region_pred}\n\n"
            f"{resumen.get('interpretacion_global', '')}\n\n"
            f"{resumen.get('interpretacion_snr', '')}\n\n"
            f"{resumen.get('analisis_regional', {}).get('interpretacion_regional', '')}\n\n"
            "Ecuaciones usadas\n"
            "-----------------\n"
            "% ponderado = % atípicos + 0.5 × % sospechosos\n"
            "% sin ponderar = ((N_atípicos + N_sospechosos) / N_total) × 100\n"
            "SNR = P_señal / P_ruido\n"
            "SNR_dB = 10 × log10(SNR)\n"
            "P_señal = potencia integrada entre 1 y 30 Hz\n"
            "P_ruido = potencia integrada entre 40 y 100 Hz\n\n"
            "Canales detectados\n"
            "------------------\n"
            f"Canales atípicos: {canales_atipicos}\n\n"
            f"Canales sospechosos: {canales_sospechosos}\n\n"
            "Nota técnica\n"
            "------------\n"
            "La relación señal/ruido (SNR) compara la potencia de la señal EEG útil con la potencia del ruido de fondo. "
            "En este análisis, la señal útil se estima integrando la potencia entre 1 y 30 Hz, mientras que el ruido se "
            "estima entre 40 y 100 Hz. Un valor alto de SNR indica una señal más limpia y de mayor calidad.\n\n"
            "Los canales sospechosos se ponderan con menor peso que los atípicos fuertes porque representan alertas "
            "moderadas y no alteraciones técnicas robustas. Esta métrica permite evaluar si la muestra es técnicamente "
            "confiable antes de interpretar bandas cerebrales, ratios espectrales o mapas topográficos.\n"
        )

    def _dibujar_dashboard_calidad(self, resumen):
        if self.fig_calidad is None or self.canvas_calidad is None:
            return

        if hasattr(self, "widget_calidad") and self.widget_calidad is not None:
            w = self.widget_calidad.winfo_width()
            h = self.widget_calidad.winfo_height()
            if w > 300 and h > 220:
                dpi = self.fig_calidad.get_dpi()
                self.fig_calidad.set_size_inches(w / dpi, h / dpi, forward=False)

        self.fig_calidad.clear()
        gs = self.fig_calidad.add_gridspec(
            1, 2,
            width_ratios=[1, 1],
            left=0.05,
            right=0.95,
            bottom=0.08,
            top=0.88,
            wspace=0.22
        )
        ax_pie = self.fig_calidad.add_subplot(gs[0, 0])
        ax_mapa = self.fig_calidad.add_subplot(gs[0, 1])
        ax_pie.set_aspect("equal")
        ax_mapa.set_aspect("equal")

        valores = [
            resumen.get("n_normales", 0),
            resumen.get("n_sospechosos", 0),
            resumen.get("n_atipicos", 0)
        ]

        etiquetas = [
            f"Normales\n{float(resumen.get('porcentaje_normales', 0.0)):.1f}%",
            f"Sospechosos\n{float(resumen.get('porcentaje_sospechosos', 0.0)):.1f}%",
            f"Atípicos\n{float(resumen.get('porcentaje_atipicos', 0.0)):.1f}%"
        ]

        if sum(valores) == 0:
            ax_pie.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax_pie.axis("off")
        else:
            colores = ["#22C55E", "#F59E0B", "#DC2626"]
            ax_pie.pie(
                valores,
                labels=etiquetas,
                autopct=None,
                colors=colores,
                startangle=90,
                labeldistance=1.08,
                radius=0.82,
                wedgeprops={"width": 0.36, "edgecolor": "white"},
                textprops={"fontsize": 8}
            )
            ax_pie.text(0, 0, "Calidad\nEEG", ha="center", va="center", fontsize=10, weight="bold")
            ax_pie.set_title("Distribución global", fontsize=9)

        self._dibujar_mapa_calidad_electrodos(ax_mapa, resumen)

        self.canvas_calidad.draw()

    def _actualizar_panel_izquierdo_calidad(self, resumen):
        metricas = self._metricas_calidad_resumen(resumen)
        estado = metricas["estado"]
        pct = metricas["pct_ponderado"]
        snr_db = metricas["snr_db"]
        region = metricas["region"]

        atip = self._formatear_lista_canales_compacta(resumen.get("canales_atipicos", []))
        sosp = self._formatear_lista_canales_compacta(resumen.get("canales_sospechosos", []))

        self.lbl_calidad_estado.config(text=f"Estado global: {estado}")
        self.lbl_calidad_estado.config(foreground=self._color_estado_calidad(estado))
        self.lbl_calidad_pct.config(text=f"Índice técnico ponderado: {self._formatear_pct(pct)}")
        if hasattr(self, "lbl_calidad_pct_detalle"):
            self.lbl_calidad_pct_detalle.config(
                text=(
                    f"Atípicos: {self._formatear_pct(metricas['pct_atipicos'])} | "
                    f"Sospechosos: {self._formatear_pct(metricas['pct_sospechosos'])}"
                )
            )
        self.lbl_calidad_snr.config(text=f"SNR global: {snr_db:.2f} dB")
        self.lbl_calidad_region.config(text=f"Región predominante: {region}")
        self.lbl_calidad_atipicos.config(text=f"Canales atípicos: {atip}")
        self.lbl_calidad_sospechosos.config(text=f"Canales sospechosos: {sosp}")
        self.lbl_calidad_atipicos.config(text=f"Canales atípicos:\n{atip}")
        self.lbl_calidad_sospechosos.config(text=f"Canales sospechosos:\n{sosp}")
        self._actualizar_resumen_visual_calidad(resumen)

    def mostrar_dashboard_calidad_senal(self, seleccionar_tab=True, mostrar_warning=False):
        resumen = self.cargar_resumen_calidad(mostrar_warning=mostrar_warning)
        self._actualizar_bloques_resumen_calidad()

        if seleccionar_tab and hasattr(self, "tabs_analisis"):
            try:
                self.panel_grafica.pack_forget()
                self.panel_analisis.pack(fill=tk.BOTH, expand=True)
                self._mostrar_marco_visor_grafica(False)
                self._mostrar_tabs_grafica(False)
                self._mostrar_panel_izquierdo("analisis")
                self.tabs_analisis.select(self.tab_calidad_senal)
                self._mostrar_panel_izq_analisis_actual()
            except Exception:
                pass

        if resumen is None:
            self._reset_panel_izquierdo_calidad(
                "Procesando calidad técnica..."
            )
            if self.fig_calidad is not None:
                self.fig_calidad.clear()
                ax = self.fig_calidad.add_subplot(111)
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    "⏳ Calculando Calidad Técnica de la señal...\nPor favor espere un momento.",
                    ha="center", va="center", fontsize=11, color="#1a5276"
                )
                self.canvas_calidad.draw()
                self.canvas_calidad.flush_events()
            return

        self._actualizar_panel_izquierdo_calidad(resumen)
        self._dibujar_dashboard_calidad(resumen)
        self._actualizar_scrollregion_izquierda()
    
    def _cargar_er_filtrado_cache(self):
        """
        Carga energía relativa por canal (FILTRADO) desde cache.
        Espera carpeta:
            <cache_ultimo>/energia_relativa_filtrado/Delta.npy (etc)
        Cada .npy debe ser un vector de 64 (porcentaje por canal).
        """
        self._er_filtrado = None
    
        if not self.cache_ultimo:
            return
    
        carpeta_er = os.path.join(self.cache_ultimo, "energia_relativa_filtrado")
        if not os.path.isdir(carpeta_er):
            return
    
        er = {}
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            ruta = os.path.join(carpeta_er, f"{b}.npy")
            if os.path.exists(ruta):
                er[b] = np.load(ruta, allow_pickle=True)
    
        # Validación mínima
        if all(b in er for b in ["Delta", "Theta", "Alfa", "Beta"]):
            self._er_filtrado = er

    def _cargar_potencia_relativa_welch_cache(self):
        """
        Carga potencia relativa Welch (%) desde cache/analisis_welch.
        """
        self._pot_rel_welch = None
        self._resumen_welch = None

        if not self.cache_ultimo:
            print("\n[DIAGNÓSTICO CACHÉ]: self.cache_ultimo no está definido.")
            return False

        carpeta = os.path.join(self.cache_ultimo, "analisis_welch")
        ruta_npz = os.path.join(carpeta, "pot_rel_welch.npz")
        ruta_json = os.path.join(carpeta, "pot_rel_promedios.json")

        if not os.path.exists(ruta_npz):
            # ESTA ES LA LÍNEA QUE SE ACTIVA: El archivo .npz no existe en el disco
            print(f"\n[DIAGNÓSTICO CACHÉ]: No se encontró el archivo {ruta_npz}.")
            print("El pipeline no guardó pot_rel_welch.npz al procesar la señal.")
            return False

        try:
            datos = {}
            with np.load(ruta_npz, allow_pickle=False) as z:
                for b in ["Delta", "Theta", "Alfa", "Beta"]:
                    if b not in z:
                        print(f"\n[DIAGNÓSTICO CACHÉ]: Falta la banda '{b}' en el archivo .npz.")
                        return False
                    datos[b] = np.asarray(z[b], dtype=np.float64)

            resumen = None
            if os.path.exists(ruta_json):
                with open(ruta_json, "r", encoding="utf-8") as f:
                    resumen = json.load(f)

            self._pot_rel_welch = datos
            self._resumen_welch = resumen
            return True
        except Exception as e:
            import traceback
            print(f"\n[ERROR CRÍTICO AL LEER .NPZ]: {e}")
            traceback.print_exc()
            self.log(f"No pude cargar potencia relativa Welch: {e}")
            self._pot_rel_welch = None
            self._resumen_welch = None
            return False

    def cargar_potencia_relativa_desde_cache(self):
        """
        Alias claro para la vista: solo lee potencia relativa Welch desde cache.
        No recalcula nada en la interfaz.
        """
        return self._cargar_potencia_relativa_welch_cache()

    def _promedios_globales_potencia_welch(self):
        if self._pot_rel_welch is None:
            return {}

        if isinstance(self._resumen_welch, dict):
            proms = self._resumen_welch.get("promedios_globales", {})
            if proms:
                return proms

        return {
            b: float(np.nanmean(np.asarray(self._pot_rel_welch[b], dtype=np.float64)))
            for b in ["Delta", "Theta", "Alfa", "Beta"]
            if b in self._pot_rel_welch
        }

    def _promedios_regionales_potencia_welch(self):
        regiones_objetivo = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]

        if self._pot_rel_welch is None:
            return {}

        if isinstance(self._resumen_welch, dict):
            regiones_cache = self._resumen_welch.get("promedios_regionales", {})
            if regiones_cache:
                return {reg: regiones_cache.get(reg, {}) for reg in regiones_objetivo}

        grupos = {reg: [] for reg in regiones_objetivo}
        for i, nombre in enumerate(self.nombres_canales):
            reg = self._region_de_canal_ui(self._nombre_simple(nombre))
            if reg in grupos:
                grupos[reg].append(i)

        proms = {}
        for reg, idxs in grupos.items():
            proms[reg] = {}
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                arr = np.asarray(self._pot_rel_welch.get(b, []), dtype=np.float64)
                if len(idxs) and arr.size > max(idxs):
                    proms[reg][b] = float(np.nanmean(arr[idxs]))
                else:
                    proms[reg][b] = float("nan")
        return proms

    def _reset_panel_izquierdo_potencia_relativa(self, mensaje="Sin datos cargados."):
        if not hasattr(self, "lbl_estado_welch_izq"):
            return

        self.lbl_estado_welch_izq.config(text=mensaje)
        self._actualizar_tabla_welch_global()
        self._actualizar_tabla_regiones_welch()
        self.lbl_welch_sel_canal.config(text="Canal seleccionado: --")
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            self.lbl_welch_bandas[b].config(text=f"{b}: --%")

    def _actualizar_tabla_welch_global(self, promedios_globales=None):
        if not hasattr(self, "tabla_welch_global"):
            return

        for item in self.tabla_welch_global.get_children():
            self.tabla_welch_global.delete(item)

        if promedios_globales is None:
            promedios_globales = {}

        for banda in ["Delta", "Theta", "Alfa", "Beta"]:
            self._insertar_fila_treeview(
                self.tabla_welch_global,
                (banda, self._formatear_pct(promedios_globales.get(banda, None)))
            )

    def _actualizar_tabla_regiones_welch(self, promedios_regionales=None):
        if not hasattr(self, "tabla_welch_regiones"):
            return

        for item in self.tabla_welch_regiones.get_children():
            self.tabla_welch_regiones.delete(item)

        if promedios_regionales is None:
            promedios_regionales = {}

        for reg in getattr(self, "regiones_welch", ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]):
            vals = promedios_regionales.get(reg, {})
            fila = [reg]
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                fila.append(self._formatear_pct(vals.get(b, None)))
            self._insertar_fila_treeview(self.tabla_welch_regiones, fila)

    def actualizar_panel_izquierdo_potencia_relativa(self, canal_simple=None):
        """
        Alimenta la barra izquierda contextual de potencia relativa Welch:
        resumen global, resumen regional y canal seleccionado.
        """
        if not hasattr(self, "lbl_estado_welch_izq"):
            return

        if self._pot_rel_welch is None:
            ok = self.cargar_potencia_relativa_desde_cache()
            if not ok:
                self._reset_panel_izquierdo_potencia_relativa(
                    "No se encontró análisis de potencia relativa para este archivo."
                )
                return

        self.lbl_estado_welch_izq.config(text="Datos cargados desde cache Welch.")

        proms_globales = self._promedios_globales_potencia_welch()
        self._actualizar_tabla_welch_global(proms_globales)

        proms_regionales = self._promedios_regionales_potencia_welch()
        self._actualizar_tabla_regiones_welch(proms_regionales)

        canal = self._nombre_simple(canal_simple) if canal_simple else ""
        if not canal:
            canal = self.canal_potencia_welch_simple or self.canal_analisis_simple or self.canal_activo_simple
        if not canal and self.nombres_canales:
            canal = self._nombre_simple(self.nombres_canales[0])

        self.actualizar_canal_potencia_relativa(canal, refrescar_resumen=False)

    def actualizar_canal_potencia_relativa(self, canal_simple, refrescar_resumen=True):
        """
        Actualiza el bloque de canal seleccionado para potencia relativa Welch.
        Se llama desde el clic sobre el topomap Welch y desde la carga inicial.
        """
        canal_simple = self._nombre_simple(canal_simple)
        if not canal_simple:
            return

        if self._pot_rel_welch is None:
            ok = self.cargar_potencia_relativa_desde_cache()
            if not ok:
                self._reset_panel_izquierdo_potencia_relativa(
                    "No se encontró análisis de potencia relativa para este archivo."
                )
                return

        if refrescar_resumen:
            self.actualizar_panel_izquierdo_potencia_relativa(canal_simple)
            return

        idx = None
        for i, nombre in enumerate(self.nombres_canales):
            if self._nombre_simple(nombre) == canal_simple:
                idx = i
                break

        self.canal_potencia_welch_simple = canal_simple
        self.canal_analisis_simple = canal_simple

        if idx is None:
            self.lbl_welch_sel_canal.config(text=f"Canal seleccionado: {canal_simple}")
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                self.lbl_welch_bandas[b].config(text=f"{b}: --%")
            return

        self.lbl_welch_sel_canal.config(text=f"Canal seleccionado: {canal_simple}")
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            try:
                v = float(np.asarray(self._pot_rel_welch[b], dtype=np.float64)[idx])
                self.lbl_welch_bandas[b].config(text=f"{b}: {self._formatear_pct(v)}")
            except Exception:
                self.lbl_welch_bandas[b].config(text=f"{b}: --%")

        try:
            self.var_canal_nombre.set(canal_simple)
            self.combo_canal.set(canal_simple)
        except Exception:
            pass
    
    def _region_de_canal_ui(self, nombre_simple: str) -> str:
        """
        Región aproximada por prefijo (10-10/10-20).
        """
        e = (nombre_simple or "").upper().strip()
    
        if e in ("M1", "M2", "A1", "A2"):
            return "Mastoides"
        if e.startswith(("O", "PO", "CB")):
            return "Occipital"
        if e.startswith(("P", "CP")):
            return "Parietal"
        if e.startswith(("T", "FT", "TP")):
            return "Temporal"
        if e.startswith(("C", "FC")) or e.endswith("Z"):
            return "Central"
        if e.startswith(("FP", "AF", "F")):
            return "Frontal"
        return "Otro"
    
    def _color_heatmap(self, valor, p25, p50, p75):
        # colores semáforo (relleno)
        if valor <= p25:
            return "#22C55E"  # verde
        if valor <= p50:
            return "#FACC15"  # amarillo
        if valor <= p75:
            return "#FB923C"  # naranja
        return "#EF4444"      # rojo
    
    def _percentiles_banda(self, banda):
        """
        Devuelve (p25, p50, p75) de la banda actual con los 64 canales.
        Si no hay datos, devuelve None.
        """
        if self._er_filtrado is None:
            return None
        if banda not in self._er_filtrado:
            return None
    
        arr = np.asarray(self._er_filtrado[banda], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 5:
            return None
    
        p25, p50, p75 = np.percentile(arr, [25, 50, 75])
        return float(p25), float(p50), float(p75)
    

    
    
    def _coord_1010_curvada_norm(self, et: str):
        et = (et or "").upper().strip()
    
        def proyectar_dentro_circulo(x, y, R=1.0, margen=0.985):
            d2 = x*x + y*y
            lim2 = (margen*R)*(margen*R)
            if d2 <= lim2 or d2 == 0:
                return x, y
            d = math.sqrt(d2)
            esc = (margen*R) / d
            return x * esc, y * esc
    
        def s_por_prefijo_1010(pref: str) -> float:
            pref = pref.upper()
            if pref == "FP": return 0.10
            if pref == "AF": return 0.20
            if pref == "F":  return 0.31
            if pref in ("FC", "FT"): return 0.40
            if pref in ("C", "T"):   return 0.50
            if pref in ("CP", "TP"): return 0.60
            if pref == "P":  return 0.70
            if pref == "PO": return 0.80
            if pref == "O":  return 0.90
            return 0.50
    
        def lateral_por_numero(pref: str, n: int) -> float:
            pref = pref.upper()
            if n in (1, 2):
                lf = 0.22
            elif n in (3, 4):
                lf = 0.42
            elif n in (5, 6):
                lf = 0.62
            elif n in (7, 8):
                lf = 0.92
            else:
                lf = 0.50
    
            if pref in ("T", "FT", "TP") and n in (7, 8):
                lf = 0.95
            if pref in ("FP", "O") and n in (1, 2):
                lf = 0.44
            return lf
    
        def t_uniforme_en_fila(pref: str, suf: str):
            pref = pref.upper()
            suf = suf.upper()
    
            orden_por_fila = {
                "F":  ["7","5","3","1","Z","2","4","6","8"],
                "FC": ["5","3","1","Z","2","4","6"],
                "C":  ["5","3","1","Z","2","4","6"],
                "CP": ["5","3","1","Z","2","4","6"],
                "P":  ["7","5","3","1","Z","2","4","6","8"],
                "PO": ["7","5","3","Z","4","6","8"],
            }
    
            if pref not in orden_por_fila:
                return None
            orden = orden_por_fila[pref]
            if suf not in orden:
                return None
    
            i = orden.index(suf)
            mid = (len(orden) - 1) / 2.0
            return (i - mid) / mid
    
        def factor_apertura_fila(pref: str) -> float:
            pref = pref.upper()
            if pref in ("FP", "O"):
                return 1.15
            if pref in ("C", "FC", "CP"):
                return 0.82
            if pref in ("F", "P"):
                return 0.92
            if pref == "PO":
                return 1.00
            if pref in ("T", "FT", "TP"):
                return 0.97
            return 1.0
    
        if et == "M1":
            x, y = proyectar_dentro_circulo(-1.0, 0.18)
            return x, -y
        if et == "M2":
            x, y = proyectar_dentro_circulo(1.0, 0.18)
            return x, -y
        if et == "CB1":
            x, y = proyectar_dentro_circulo(-0.43, 0.86)
            return x, -y
        if et == "CB2":
            x, y = proyectar_dentro_circulo(0.43, 0.86)
            return x, -y
    
        m = re.match(r"^([A-Z]+)(\d+|Z)$", et)
        if not m:
            return None
    
        pref, suf = m.group(1), m.group(2)
    
        s = s_por_prefijo_1010(pref)
        y_base = (s - 0.5) * 2.0
    
        if pref == "F":
            y_base -= 0.02
    
        x_max = math.sqrt(max(0.0, 1.0 - y_base*y_base)) * 0.98
        if x_max < 1e-6:
            return None
    
        t = t_uniforme_en_fila(pref, suf)
        apertura = factor_apertura_fila(pref)
        lateral_global = 0.90
    
        if pref in ("T", "FT", "TP"):
            lateral_global = 0.94
        if pref in ("C", "FC", "CP"):
            lateral_global = 0.82
    
        if suf == "Z":
            x = 0.0
            y = y_base
        elif t is not None:
            x = t * x_max * lateral_global * apertura
            y = y_base
        else:
            n = int(suf)
            lf = lateral_por_numero(pref, n) * lateral_global
            signo = -1 if (n % 2 == 1) else 1
            x = signo * lf * x_max
            y = y_base
    
        if s < 0.50:
            signo_curva = -1
        elif s > 0.50:
            signo_curva = +1
        else:
            signo_curva = 0
    
        if pref in ("F", "P"):
            amp = 0.14
        elif pref in ("FC", "CP", "FT", "TP"):
            amp = 0.10
        elif pref in ("PO", "O", "FP", "AF"):
            amp = 0.08
        else:
            amp = 0.08
    
        xn = x / x_max
    
        if pref == "F":
            y = y + signo_curva * amp * 0.75 * (xn * xn)
        else:
            y = y + signo_curva * amp * (xn * xn)
    
        x, y = proyectar_dentro_circulo(x, y)
        return x, -y
    
    def _on_root_configure(self, _evt=None):
        """Detecta resize de la ventana raíz y bloquea redraws mientras dura el drag."""
        if self._cerrando:
            return
        if self._resize_job is not None:
            try:
                self.raiz.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resizing   = True
        self._resize_job = self.raiz.after(450, self._on_resize_end)

    def _on_resize_end(self):
        """Se dispara 450ms después del último evento de resize."""
        self._resize_job = None
        self._resizing   = False
        # Un único redraw al soltar la ventana
        self._programar_redibujo_welch()
        self._programar_redibujo_ratios()
        self._programar_redibujo_topomap()

    def _on_topo_configure(self, event=None):
        if self._cerrando or self._resizing:
            return
        self._programar_redibujo_topomap()

    def _on_welch_configure(self, event=None):
        if self._cerrando or self._resizing:
            return
        self._programar_redibujo_welch()

    def _on_ratios_configure(self, event=None):
        if self._cerrando or self._resizing:
            return
        self._programar_redibujo_ratios()

    def _toggle_etiquetas_topo(self):
        if self._cerrando:
            return
    
        if self._after_topo_resize is not None:
            try:
                self.raiz.after_cancel(self._after_topo_resize)
            except Exception:
                pass
            self._after_topo_resize = None
    
        self.raiz.after(20, self._redibujar_topomap_si_visible)
        self.raiz.after(20, self._redibujar_welch_si_visible)
        self.raiz.after(20, self._redibujar_ratios_si_visible)

    def _actualizar_resumen_welch(self):
        if self._pot_rel_welch is None:
            self._reset_panel_izquierdo_potencia_relativa(
                "No se encontró análisis de potencia relativa para este archivo."
            )
            return

        self.actualizar_panel_izquierdo_potencia_relativa()

    def _dibujar_topomap_welch(self):
        if self.fig_welch is None or self.canvas_welch is None:
            return

        if self._welch_redraw_en_progreso:
            return

        if hasattr(self, "_loading") and not getattr(self, "_resizing", False):
            self._loading.show()
            
        self._welch_redraw_en_progreso = True
        try:
            if self._pot_rel_welch is None:
                ok = self._cargar_potencia_relativa_welch_cache()
                if not ok:
                    self.fig_welch.clear()
                    ax = self.fig_welch.add_subplot(111)
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5,
                        "⏳ Procesando análisis Welch y Calidad Técnica...\nPor favor espere un momento.",
                        ha="center", va="center", fontsize=11, color="#1a5276"
                    )
                    self.canvas_welch.draw()
                    self.canvas_welch.flush_events()  
                    return

            self._actualizar_resumen_welch()

            if hasattr(self, "widget_welch"):
                w = self.widget_welch.winfo_width()
                h = self.widget_welch.winfo_height()

                if w < 300 or h < 300:
                    return

                dpi = self.fig_welch.get_dpi()
                self.fig_welch.set_size_inches(w / dpi, h / dpi, forward=False)

            axes = self._dibujar_topomaps_valores(
                self.fig_welch,
                self._pot_rel_welch,
                self._welch_topo_xy,
                self.cbar_welch
            )
            self._ax_welch_to_band = {ax: banda for banda, ax in axes.items()}
            self.canvas_welch.draw()
        finally:
            self._welch_redraw_en_progreso = False
            if hasattr(self, "_loading"):
                self.raiz.after(120, self._loading.hide)

    def _valores_topomap_ratios_actual(self):
        if self._ratio_personalizado_activo:
            self._ratio_personalizado = self.calcular_ratio_personalizado_desde_pot_rel()
            if self._ratio_personalizado is None:
                return None, None, None
            nombre = self._nombre_ratio_personalizado()
            return {nombre: self._ratio_personalizado}, [nombre], {nombre: f"Personalizada: {nombre}"}

        if self._ratios_principales is None:
            ok = self.cargar_ratios_bandas_principales_desde_cache()
            if not ok:
                return None, None, None

        labels = self._ratios_labels()
        valores = {}
        nombres = []
        titulos = {}
        for key, label in labels.items():
            if key in self._ratios_principales:
                valores[label] = self._ratios_principales[key]
                nombres.append(label)
                titulos[label] = label
        return valores, nombres, titulos

    def dibujar_topomap_ratio_personalizado(self):
        self._ratio_personalizado_activo = True
        self._dibujar_topomap_ratios()

    def _dibujar_topomap_ratios(self):
        if self.fig_ratios is None or self.canvas_ratios is None:
            return

        if self._ratios_redraw_en_progreso:
            return

        if hasattr(self, "_loading") and not getattr(self, "_resizing", False):
            self._loading.show()
        self._ratios_redraw_en_progreso = True
        try:
            if self._ratios_principales is None:
                ok = self.cargar_ratios_bandas_principales_desde_cache()
                if not ok:
                    self._reset_panel_izquierdo_ratios(
                        "Calculando proporción de bandas..."
                    )
                    self.fig_ratios.clear()
                    ax = self.fig_ratios.add_subplot(111)
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5,
                        "⏳ Calculando proporción de bandas...\nPor favor espere un momento.",
                        ha="center", va="center", fontsize=11, color="#1a5276"
                    )
                    self.canvas_ratios.draw()
                    self.canvas_ratios.flush_events()
                    return

            self.actualizar_panel_izquierdo_ratios()

            if hasattr(self, "widget_ratios"):
                w = self.widget_ratios.winfo_width()
                h = self.widget_ratios.winfo_height()

                if w < 300 or h < 300:
                    return

                dpi = self.fig_ratios.get_dpi()
                self.fig_ratios.set_size_inches(w / dpi, h / dpi, forward=False)

            valores, nombres, titulos = self._valores_topomap_ratios_actual()
            if not valores or not nombres:
                return

            axes = self._dibujar_topomaps_valores(
                self.fig_ratios,
                valores,
                self._ratios_topo_xy,
                self.cbar_ratios,
                nombres=nombres,
                titulos=titulos,
                cbar_label="Razón"
            )
            self._ax_ratios_to_nombre = {ax: nombre for nombre, ax in axes.items()}
            self.canvas_ratios.draw()
        finally:
            self._ratios_redraw_en_progreso = False
            if hasattr(self, "_loading"):
                self.raiz.after(120, self._loading.hide)

    def _dibujar_topomap_analisis(self):
        if self.fig_topo is None or self.canvas_topo is None:
            return

        if self._topo_redraw_en_progreso:
            return

        if hasattr(self, "_loading") and not getattr(self, "_resizing", False):
            self._loading.show()

        self._topo_redraw_en_progreso = True
        try:
            if self._er_filtrado is None:
                self._cargar_er_filtrado_cache()
    
            if hasattr(self, "widget_topo"):
                w = self.widget_topo.winfo_width()
                h = self.widget_topo.winfo_height()
            
                if w < 300 or h < 300:
                    return
            
                dpi = self.fig_topo.get_dpi()
                self.fig_topo.set_size_inches(w / dpi, h / dpi, forward=False)
            
                self._ultimo_size_topo = (w, h)
    
            axes, caxes = self._crear_grilla_topomaps()
    
            bandas = ["Delta", "Theta", "Alfa", "Beta"]
            self._topo_xy = {}
            self.cbar_topo = {}
    
            for banda in bandas:
                ax = axes[banda]
                cax = caxes[banda]
                self._topo_xy[banda] = {}
    
                if self._er_filtrado is None or banda not in self._er_filtrado:
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")
                    cax.axis("off")
                    continue
    
                valores = np.asarray(self._er_filtrado[banda], dtype=float)
    
                xs, ys, zs = [], [], []
                for ch, val in zip(self.nombres_canales, valores):
                    et = self._nombre_simple(ch)
                    p = self._coord_1010_curvada_norm(et)
                    if p is None:
                        continue
                    x, y = p
                    xs.append(x)
                    ys.append(y)
                    zs.append(float(val))
                    self._topo_xy[banda][et] = (x, y)
    
                xs = np.array(xs, dtype=float)
                ys = np.array(ys, dtype=float)
                zs = np.array(zs, dtype=float)

                if xs.size < 4:
                    ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")
                    cax.axis("off")
                    continue
    
                gx = np.linspace(-1.05, 1.05, 250)
                gy = np.linspace(-1.05, 1.05, 250)
                XI, YI = np.meshgrid(gx, gy)
    
                mascara = (XI**2 + YI**2) <= 1.0**2
    
                puntos = np.column_stack([xs, ys])
                try:
                    ZI = griddata(
                        points=puntos,
                        values=zs,
                        xi=(XI, YI),
                        method="cubic"
                    )
                except Exception:
                    try:
                        ZI = griddata(
                            points=puntos,
                            values=zs,
                            xi=(XI, YI),
                            method="linear"
                        )
                    except Exception:
                        ZI = None

                ZI_near = griddata(
                    points=puntos,
                    values=zs,
                    xi=(XI, YI),
                    method="nearest"
                )
                if ZI is None:
                    ZI = ZI_near
    
                ZI = np.where(np.isnan(ZI), ZI_near, ZI)
                ZI_suave = gaussian_filter(ZI, sigma=2.2)
                ZI_suave[~mascara] = np.nan
    
                vmin = float(np.nanmin(ZI_suave))
                vmax = float(np.nanmax(ZI_suave))
    
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")
                    cax.axis("off")
                    continue
    
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6
    
                niveles = np.linspace(vmin, vmax, 24)
    
                im = ax.contourf(
                    XI, YI, ZI_suave,
                    levels=niveles,
                    cmap="turbo"
                )
    
                cbar = self.fig_topo.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=6)
                cbar.ax.yaxis.set_major_formatter(self._fmt_cbar_pct)
                cbar.set_label("%", fontsize=7)
                self.cbar_topo[banda] = cbar
    
                ax.contour(
                    XI, YI, ZI_suave,
                    levels=10,
                    colors="k",
                    linewidths=0.20,
                    alpha=0.18
                )
    
                cabeza = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.8)
                ax.add_patch(cabeza)
                ax.plot([-0.08, 0.0, 0.08], [1.00, 1.10, 1.00], color="black", lw=1.8)
    
                t = np.linspace(-0.9, 0.9, 80)
                ax.plot(-1.0 - 0.015*np.cos(t), 0.20*np.sin(t), color="black", lw=2.0, clip_on=True)
                ax.plot( 1.0 + 0.015*np.cos(t), 0.20*np.sin(t), color="black", lw=2.0, clip_on=True)
    
                ax.scatter(xs, ys, s=12, c="black", zorder=3)
    
                if self.var_etiquetas_topo.get():
                    for ch in self.nombres_canales:
                        et = self._nombre_simple(ch)
                        if et in self._topo_xy[banda]:
                            x, y = self._topo_xy[banda][et]
                            ax.text(x, y + 0.022, et, fontsize=6, ha="center", va="bottom")
    
                ax.set_title(banda, fontsize=10)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-1.08, 1.08)
                ax.set_ylim(-1.08, 1.16)
                ax.set_anchor("C")
                ax.margins(0)
                ax.axis("off")
    
            #self.canvas_topo.draw_idle()
            self.canvas_topo.draw()

        finally:
            self._topo_redraw_en_progreso = False
            if hasattr(self, "_loading"):
                self.raiz.after(120, self._loading.hide)
            
    

            
    def _on_motion_topomap(self, event):
        if event.inaxes is None or not self._topo_xy:
            self._tooltip_ocultar()
            return
    
        banda = self._ax_to_band.get(event.inaxes, None)
        if banda is None:
            self._tooltip_ocultar()
            return
    
        if event.xdata is None or event.ydata is None:
            self._tooltip_ocultar()
            return
    
        pos = self._topo_xy.get(banda, {})
        if not pos:
            self._tooltip_ocultar()
            return
    
        mx, my = float(event.xdata), float(event.ydata)
    
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et
    
        if mejor is None or mejor_d2 > 0.015:
            self._tooltip_ocultar()
            return
    
        idx = None
        for i, n in enumerate(self.nombres_canales):
            if self._nombre_simple(n) == mejor:
                idx = i
                break
    
        if idx is None or self._er_filtrado is None or banda not in self._er_filtrado:
            self._tooltip_ocultar()
            return
    
        v = float(self._er_filtrado[banda][idx])
        txt = f"{mejor}\n{banda}: {self._formatear_pct(v)}"
    
        try:
            x_root = self.raiz.winfo_pointerx()
            y_root = self.raiz.winfo_pointery()
            self._tooltip_mostrar(x_root, y_root, txt)
        except Exception:
            pass
    
    
    def _on_click_topomap(self, event):
        if event.inaxes is None or not self._topo_xy:
            return
    
        banda = self._ax_to_band.get(event.inaxes, None)
        if banda is None:
            return
    
        if event.xdata is None or event.ydata is None:
            return
    
        pos = self._topo_xy.get(banda, {})
        if not pos:
            return
    
        mx, my = float(event.xdata), float(event.ydata)
    
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et
    
        if mejor is None or mejor_d2 > 0.020:
            return
    
        self._click_canal_analisis(mejor)

    def _on_click_topomap_welch(self, event):
        if event.inaxes is None or not self._welch_topo_xy:
            return

        banda = self._ax_welch_to_band.get(event.inaxes, None)
        if banda is None or event.xdata is None or event.ydata is None:
            return

        pos = self._welch_topo_xy.get(banda, {})
        if not pos:
            return

        mx, my = float(event.xdata), float(event.ydata)
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et

        if mejor is None or mejor_d2 > 0.020:
            return

        self.actualizar_canal_potencia_relativa(mejor)

    def _dibujar_topomaps_valores(
        self,
        fig,
        valores_por_banda,
        xy_store,
        cbar_store,
        nombres=None,
        titulos=None,
        cbar_label="%"
    ):
        if nombres is None:
            nombres = ["Delta", "Theta", "Alfa", "Beta"]
        if titulos is None:
            titulos = {}

        axes, caxes = self._crear_grilla_topomaps_en_fig(fig, nombres=nombres)
        xy_store.clear()
        cbar_store.clear()

        for banda in nombres:
            ax = axes[banda]
            cax = caxes[banda]
            xy_store[banda] = {}

            if valores_por_banda is None or banda not in valores_por_banda:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                cax.axis("off")
                continue

            valores = np.asarray(valores_por_banda[banda], dtype=float)

            xs, ys, zs = [], [], []
            for ch, val in zip(self.nombres_canales, valores):
                et = self._nombre_simple(ch)
                p = self._coord_1010_curvada_norm(et)
                if p is None or not np.isfinite(val):
                    continue
                x, y = p
                xs.append(x)
                ys.append(y)
                zs.append(float(val))
                xy_store[banda][et] = (x, y)

            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            zs = np.array(zs, dtype=float)

            if xs.size < 4:
                ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                cax.axis("off")
                continue

            gx = np.linspace(-1.05, 1.05, 250)
            gy = np.linspace(-1.05, 1.05, 250)
            XI, YI = np.meshgrid(gx, gy)
            mascara = (XI**2 + YI**2) <= 1.0**2
            puntos = np.column_stack([xs, ys])

            try:
                ZI = griddata(points=puntos, values=zs, xi=(XI, YI), method="cubic")
            except Exception:
                try:
                    ZI = griddata(points=puntos, values=zs, xi=(XI, YI), method="linear")
                except Exception:
                    ZI = None

            ZI_near = griddata(points=puntos, values=zs, xi=(XI, YI), method="nearest")
            if ZI is None:
                ZI = ZI_near

            ZI = np.where(np.isnan(ZI), ZI_near, ZI)
            ZI_suave = gaussian_filter(ZI, sigma=2.2)
            ZI_suave[~mascara] = np.nan

            vmin = float(np.nanmin(ZI_suave))
            vmax = float(np.nanmax(ZI_suave))

            if not np.isfinite(vmin) or not np.isfinite(vmax):
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                cax.axis("off")
                continue

            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-6

            niveles = np.linspace(vmin, vmax, 24)
            im = ax.contourf(XI, YI, ZI_suave, levels=niveles, cmap="turbo")

            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.set_major_formatter(self._fmt_cbar_pct)
            cbar.set_label(cbar_label, fontsize=7)
            cbar_store[banda] = cbar

            ax.contour(
                XI, YI, ZI_suave,
                levels=10,
                colors="k",
                linewidths=0.20,
                alpha=0.18
            )

            cabeza = plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1.8)
            ax.add_patch(cabeza)
            ax.plot([-0.08, 0.0, 0.08], [1.00, 1.10, 1.00], color="black", lw=1.8)

            t = np.linspace(-0.9, 0.9, 80)
            ax.plot(-1.0 - 0.015*np.cos(t), 0.20*np.sin(t), color="black", lw=2.0, clip_on=True)
            ax.plot( 1.0 + 0.015*np.cos(t), 0.20*np.sin(t), color="black", lw=2.0, clip_on=True)

            ax.scatter(xs, ys, s=12, c="black", zorder=3)

            if self.var_etiquetas_topo.get():
                for ch in self.nombres_canales:
                    et = self._nombre_simple(ch)
                    if et in xy_store[banda]:
                        x, y = xy_store[banda][et]
                        ax.text(x, y + 0.022, et, fontsize=6, ha="center", va="bottom")

            ax.set_title(titulos.get(banda, banda), fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.16)
            ax.set_anchor("C")
            ax.margins(0)
            ax.axis("off")

        return axes

    def _crear_grilla_topomaps_en_fig(self, fig, nombres=None):
        """
        Crea una grilla estable:
        mapa | barra | mapa | barra
        mapa | barra | mapa | barra
        """
        if nombres is None:
            nombres = ["Delta", "Theta", "Alfa", "Beta"]

        fig.clear()

        if len(nombres) == 1:
            gs = fig.add_gridspec(
                1, 2,
                width_ratios=[1.0, 0.055],
                left=0.18,
                right=0.82,
                bottom=0.08,
                top=0.90,
                wspace=0.08
            )
            axes = {nombres[0]: fig.add_subplot(gs[0, 0])}
            caxes = {nombres[0]: fig.add_subplot(gs[0, 1])}
            return axes, caxes
    
        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[1.0, 0.06, 1.0, 0.06],
            height_ratios=[1.0, 1.0],
            # Desplaza la grilla completa un poco a la izquierda
            # sin cambiar el orden ni las proporciones internas.
            left=0.03,
            right=0.91,
            bottom=0.07,
            top=0.93,
            wspace=0.22,
            hspace=0.28
        )
    
        posiciones_axes = [(0, 0), (0, 2), (1, 0), (1, 2)]
        posiciones_caxes = [(0, 1), (0, 3), (1, 1), (1, 3)]
        axes = {}
        caxes = {}
        for nombre, pos_ax, pos_cax in zip(nombres[:4], posiciones_axes, posiciones_caxes):
            axes[nombre] = fig.add_subplot(gs[pos_ax[0], pos_ax[1]])
            caxes[nombre] = fig.add_subplot(gs[pos_cax[0], pos_cax[1]])
    
        return axes, caxes

    def _crear_grilla_topomaps(self):
        axes, caxes = self._crear_grilla_topomaps_en_fig(self.fig_topo)
    
        self.ax_topo_delta = axes["Delta"]
        self.ax_topo_theta = axes["Theta"]
        self.ax_topo_alfa  = axes["Alfa"]
        self.ax_topo_beta  = axes["Beta"]
    
        self._ax_to_band = {
            self.ax_topo_delta: "Delta",
            self.ax_topo_theta: "Theta",
            self.ax_topo_alfa:  "Alfa",
            self.ax_topo_beta:  "Beta",
        }
    
        return axes, caxes

    
    def _tooltip_mostrar(self, x_root, y_root, texto):
        if self._tt_win is None or not self._tt_win.winfo_exists():
            self._tt_win = tk.Toplevel(self.raiz)
            self._tt_win.overrideredirect(True)
            self._tt_win.attributes("-topmost", True)
            self._tt_lbl = ttk.Label(self._tt_win, text=texto, padding=(8, 4))
            self._tt_lbl.pack()
        else:
            self._tt_lbl.config(text=texto)
    
        # posición del tooltip
        self._tt_win.geometry(f"+{x_root+12}+{y_root+12}")
    
    def _tooltip_ocultar(self):
        self._tt_last = None
        if self._tt_win is not None and self._tt_win.winfo_exists():
            self._tt_win.destroy()
        self._tt_win = None
        self._tt_lbl = None
    
        
    def _on_motion_analisis_multi(self, event, banda):
        if not hasattr(self, "_pos_1010_multi"):
            self._tooltip_ocultar()
            return
    
        if self._er_filtrado is None:
            self._cargar_er_filtrado_cache()
            if self._er_filtrado is None:
                self._tooltip_ocultar()
                return
    
        if banda not in self._er_filtrado:
            self._tooltip_ocultar()
            return
    
        pos = self._pos_1010_multi.get(banda, {})
        if not pos:
            self._tooltip_ocultar()
            return
    
        mx, my = event.x, event.y
        mejor = None
        mejor_d2 = 10**9
    
        for et, (x, y) in pos.items():
            dx = mx - x
            dy = my - y
            d2 = dx * dx + dy * dy
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et
    
        if mejor is None or mejor_d2 > (22 * 22):
            self._tooltip_ocultar()
            return
    
        idx = None
        for i, n in enumerate(self.nombres_canales):
            if self._nombre_simple(n) == mejor:
                idx = i
                break
    
        if idx is None:
            self._tooltip_ocultar()
            return
    
        v = float(self._er_filtrado[banda][idx])
        texto = f"{mejor} | {banda}: {self._formatear_pct(v)}"
        self._tooltip_mostrar(event.x_root, event.y_root, texto)

    def _on_motion_topomap_welch(self, event):
        if event.inaxes is None or not self._welch_topo_xy:
            self._tooltip_ocultar()
            return

        banda = self._ax_welch_to_band.get(event.inaxes, None)
        if banda is None:
            self._tooltip_ocultar()
            return

        if event.xdata is None or event.ydata is None:
            self._tooltip_ocultar()
            return

        pos = self._welch_topo_xy.get(banda, {})
        if not pos or self._pot_rel_welch is None or banda not in self._pot_rel_welch:
            self._tooltip_ocultar()
            return

        mx, my = float(event.xdata), float(event.ydata)
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et

        if mejor is None or mejor_d2 > 0.020:
            self._tooltip_ocultar()
            return

        idx = None
        for i, ch in enumerate(self.nombres_canales):
            if self._nombre_simple(ch) == mejor:
                idx = i
                break

        if idx is None:
            self._tooltip_ocultar()
            return

        try:
            v = float(self._pot_rel_welch[banda][idx])
        except Exception:
            self._tooltip_ocultar()
            return

        self._tooltip_mostrar(event.x_root, event.y_root, f"{mejor} | {banda} Welch: {self._formatear_pct(v)}")

    def _valor_ratio_por_nombre(self, nombre_ratio):
        if self._ratio_personalizado_activo and self._ratio_personalizado is not None:
            if nombre_ratio == self._nombre_ratio_personalizado():
                return self._ratio_personalizado

        labels = self._ratios_labels()
        for key, label in labels.items():
            if label == nombre_ratio and self._ratios_principales is not None:
                return self._ratios_principales.get(key)
        return None

    def _on_motion_topomap_ratios(self, event):
        if event.inaxes is None or not self._ratios_topo_xy:
            self._tooltip_ocultar()
            return

        nombre_ratio = self._ax_ratios_to_nombre.get(event.inaxes, None)
        if nombre_ratio is None or event.xdata is None or event.ydata is None:
            self._tooltip_ocultar()
            return

        pos = self._ratios_topo_xy.get(nombre_ratio, {})
        valores = self._valor_ratio_por_nombre(nombre_ratio)
        if not pos or valores is None:
            self._tooltip_ocultar()
            return

        mx, my = float(event.xdata), float(event.ydata)
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et

        if mejor is None or mejor_d2 > 0.020:
            self._tooltip_ocultar()
            return

        idx = None
        for i, ch in enumerate(self.nombres_canales):
            if self._nombre_simple(ch) == mejor:
                idx = i
                break

        if idx is None:
            self._tooltip_ocultar()
            return

        try:
            v = float(np.asarray(valores, dtype=np.float64)[idx])
        except Exception:
            self._tooltip_ocultar()
            return

        self._tooltip_mostrar(event.x_root, event.y_root, f"{mejor} | {nombre_ratio}: {self._formatear_ratio(v)}")

    def _on_click_topomap_ratios(self, event):
        if event.inaxes is None or not self._ratios_topo_xy:
            return

        nombre_ratio = self._ax_ratios_to_nombre.get(event.inaxes, None)
        if nombre_ratio is None or event.xdata is None or event.ydata is None:
            return

        pos = self._ratios_topo_xy.get(nombre_ratio, {})
        if not pos:
            return

        mx, my = float(event.xdata), float(event.ydata)
        mejor = None
        mejor_d2 = 1e9
        for et, (x, y) in pos.items():
            d2 = (mx - x)**2 + (my - y)**2
            if d2 < mejor_d2:
                mejor_d2 = d2
                mejor = et

        if mejor is None or mejor_d2 > 0.020:
            return

        self.actualizar_canal_ratios(mejor)

    def _distancia_punto_segmento(self, px, py, x1, y1, x2, y2):
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        cx = x1 + t * vx
        cy = y1 + t * vy
        return math.hypot(px - cx, py - cy)

    def _programar_redibujo_topomap(self, _evt=None):
        if self._cerrando:
            return
    
        if self._after_topo_resize is not None:
            try:
                self.raiz.after_cancel(self._after_topo_resize)
            except Exception:
                pass
            self._after_topo_resize = None
    
        self._after_topo_resize = self.raiz.after(200, self._redibujar_topomap_si_visible)

    def _programar_redibujo_welch(self, _evt=None):
        if self._cerrando:
            return

        if self._after_welch_resize is not None:
            try:
                self.raiz.after_cancel(self._after_welch_resize)
            except Exception:
                pass
            self._after_welch_resize = None

        self._after_welch_resize = self.raiz.after(200, self._redibujar_welch_si_visible)

    def _programar_redibujo_ratios(self, _evt=None):
        if self._cerrando:
            return

        if self._after_ratios_resize is not None:
            try:
                self.raiz.after_cancel(self._after_ratios_resize)
            except Exception:
                pass
            self._after_ratios_resize = None

        self._after_ratios_resize = self.raiz.after(200, self._redibujar_ratios_si_visible)

    def _redibujar_topomap_si_visible(self):
        self._after_topo_resize = None
    
        try:
            if not hasattr(self, "panel_analisis"):
                return
            if not self.panel_analisis.winfo_ismapped():
                return
            if self._tab_analisis_actual() != "energia":
                return
            if not hasattr(self, "widget_topo"):
                return
    
            w = self.widget_topo.winfo_width()
            h = self.widget_topo.winfo_height()
    
            if w < 300 or h < 300:
                return
    
            self._dibujar_topomap_analisis()
    
        except Exception as e:
            self.log(f"Error topomap: {e}")

    def _redibujar_welch_si_visible(self):
        self._after_welch_resize = None

        try:
            if not hasattr(self, "panel_analisis"):
                return
            if not self.panel_analisis.winfo_ismapped():
                return
            if self._tab_analisis_actual() != "welch":
                return
            if not hasattr(self, "widget_welch"):
                return

            w = self.widget_welch.winfo_width()
            h = self.widget_welch.winfo_height()

            if w < 300 or h < 300:
                return

            self._dibujar_topomap_welch()
        except Exception as e:
            self.log(f"Error topomap Welch: {e}")

    def _redibujar_ratios_si_visible(self):
        self._after_ratios_resize = None

        try:
            if not hasattr(self, "panel_analisis"):
                return
            if not self.panel_analisis.winfo_ismapped():
                return
            if self._tab_analisis_actual() != "ratios":
                return
            if not hasattr(self, "widget_ratios"):
                return

            w = self.widget_ratios.winfo_width()
            h = self.widget_ratios.winfo_height()

            if w < 300 or h < 300:
                return

            self._dibujar_topomap_ratios()
        except Exception as e:
            self.log(f"Error topomap ratios: {e}")

    def _click_canal_analisis(self, canal_simple: str):
        """
        Al hacer clic en un canal:
        - muestra % por banda del canal (FILTRADO)
        - muestra región + promedio regional por bandas
        """
        canal_simple = self._nombre_simple(canal_simple)
        self.lbl_sel_canal.config(text=f"Canal: {canal_simple}")
    
        if self._er_filtrado is None:
            self._cargar_er_filtrado_cache()
    
        # si aún no hay datos
        if self._er_filtrado is None:
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                self.lbl_bandas[b].config(text=f"{b}: (no disponible)")
                self.lbl_region_bandas[b].config(text=f"{b} (región): (no disponible)")
            self.lbl_region.config(text="Región: --")
            return
    
        # índice del canal dentro de nombres_canales
        idx = None
        for i, n in enumerate(self.nombres_canales):
            if self._nombre_simple(n) == canal_simple:
                idx = i
                break
    
        if idx is None:
            for b in ["Delta", "Theta", "Alfa", "Beta"]:
                self.lbl_bandas[b].config(text=f"{b}: --%")
                self.lbl_region_bandas[b].config(text=f"{b} (región): --%")
            self.lbl_region.config(text="Región: --")
            return
    
        # valores del canal
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            v = float(self._er_filtrado[b][idx])
            self.lbl_bandas[b].config(text=f"{b}: {self._formatear_pct(v)}")
    
        # región
        reg = self._region_de_canal_ui(canal_simple)
        self.lbl_region.config(text=f"Región: {reg}")
    
        # promedios por región (calculados en GUI)
        idxs_reg = []
        for i, n in enumerate(self.nombres_canales):
            ns = self._nombre_simple(n)
            if self._region_de_canal_ui(ns) == reg:
                idxs_reg.append(i)
    
        for b in ["Delta", "Theta", "Alfa", "Beta"]:
            arr = np.asarray(self._er_filtrado[b], dtype=np.float64)
            prom = float(np.mean(arr[idxs_reg])) if idxs_reg else float("nan")
            self.lbl_region_bandas[b].config(text=f"{b} (región): {self._formatear_pct(prom)}")
        
        # sincroniza con visor (selecciona el mismo canal en el combobox)
        self.var_canal_nombre.set(canal_simple)
        self.combo_canal.set(canal_simple)
        self.canal_analisis_simple = canal_simple

    ## =========================================================
    ## Visor: carga de señales, tiempo y redibujado
    ## =========================================================
    def _leer_fs_y_tf(self):
        """
        Intenta obtener fs desde cache:
        - fs.npy (si existe) o
        - inferido desde crudo/tiempo.npy
        """
        if not self.cache_ultimo:
            return None, None

        ruta_fs = os.path.join(self.cache_ultimo, "fs.npy")
        ruta_t = os.path.join(self.cache_ultimo, "crudo", "tiempo.npy")

        fs = None
        tf_total = None

        if os.path.exists(ruta_fs):
            try:
                arr_fs = np.asarray(np.load(ruta_fs, allow_pickle=True)).ravel()
                fs = float(arr_fs[0]) if arr_fs.size else None
                if fs is None or not np.isfinite(fs) or fs <= 0:
                    fs = None
            except Exception:
                fs = None

        if fs is None and os.path.exists(ruta_t):
            try:
                t = np.load(ruta_t, mmap_mode="r", allow_pickle=True)
                if len(t) > 2:
                    dt = float(t[1] - t[0])
                    if np.isfinite(dt) and dt > 0:
                        fs = 1.0 / dt
            except Exception as e:
                self.log(f"No pude inferir fs desde tiempo.npy: {e}")

        ruta_crudo = os.path.join(self.cache_ultimo, "crudo", "eeg_crudo.npy")
        if fs is not None and os.path.exists(ruta_crudo):
            try:
                datos = np.load(ruta_crudo, mmap_mode="r", allow_pickle=True)
                tf_total = float(datos.shape[1] / fs)
            except Exception as e:
                self.log(f"No pude leer duración desde eeg_crudo.npy: {e}")

        return fs, tf_total

    def _leer_unidad_fft_desde_cache(self):
        """
        Obtiene la unidad física asociada al registro.
        Prioriza meta.txt y, si el cache es viejo, intenta recuperarla
        desde el .dap original.
        """
        if not self.cache_ultimo:
            return None

        ruta_meta = os.path.join(self.cache_ultimo, "meta.txt")
        unidad = None
        nombre_archivo = None
        carpeta_base = None

        if os.path.exists(ruta_meta):
            try:
                with open(ruta_meta, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("data_unit="):
                            unidad = line.split("=", 1)[1].strip()
                        elif line.startswith("archivo="):
                            nombre_archivo = line.split("=", 1)[1].strip()
                        elif line.startswith("carpeta_base="):
                            carpeta_base = line.split("=", 1)[1].strip()
            except Exception as e:
                self.log(f"No pude leer unidad desde meta.txt: {e}")

        if not unidad and nombre_archivo and carpeta_base:
            try:
                ruta_dat = os.path.join(carpeta_base, nombre_archivo)
                ruta_dap = os.path.splitext(ruta_dat)[0] + ".dap"
                if os.path.exists(ruta_dap):
                    unidad = leer_dap(ruta_dap).get("data_unit")
            except Exception as e:
                self.log(f"No pude recuperar DataUnit desde .dap: {e}")

        if not unidad:
            return None

        unidad_norm = str(unidad).strip()
        unidad_cmp = unidad_norm.lower().replace("μ", "u").replace("µ", "u")
        if unidad_cmp == "uv":
            return "µV"
        return unidad_norm

    def _leer_fs_desde_meta_cache(self, fs_defecto=500.0):
        if not self.cache_ultimo:
            return float(fs_defecto)

        ruta_meta = os.path.join(self.cache_ultimo, "meta.txt")
        if not os.path.exists(ruta_meta):
            self.log("meta.txt no existe. Se usará fs=500 Hz como respaldo para el espectrograma.")
            return float(fs_defecto)

        try:
            with open(ruta_meta, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("fs="):
                        fs = float(line.split("=", 1)[1].strip())
                        if np.isfinite(fs) and fs > 0:
                            return float(fs)
                        break
        except Exception as e:
            self.log(f"No pude leer fs desde meta.txt: {e}")

        self.log("No se encontró un fs válido en meta.txt. Se usará fs=500 Hz como respaldo.")
        return float(fs_defecto)

    def _asegurar_fs_tiempo(self, datos, tiempo, fs, tf_total, fs_defecto=500.0):
        """
        Normaliza fs, duración total y vector de tiempo para las gráficas.
        Si el cache no trae tiempo válido, genera un eje temporal sintético.
        """
        n_muestras = int(datos.shape[1]) if getattr(datos, "ndim", 0) >= 2 else 0

        if fs is None:
            try:
                if len(tiempo) > 1:
                    dt_real = float(tiempo[1] - tiempo[0])
                    if np.isfinite(dt_real) and dt_real > 0:
                        fs = 1.0 / dt_real
            except Exception:
                fs = None

        if fs is None or not np.isfinite(fs) or fs <= 0:
            fs = float(fs_defecto)

        if tf_total is None or not np.isfinite(tf_total) or tf_total < 0:
            tf_total = float(n_muestras / fs) if n_muestras else 0.0

        try:
            tiempo_len = len(tiempo)
        except Exception:
            tiempo_len = 0

        if tiempo_len < n_muestras:
            tiempo = np.arange(n_muestras, dtype=np.float64) / fs

        return float(fs), float(tf_total), tiempo
    
    def _cargar_filtrado_y_tiempo(self):
        if not self.cache_ultimo:
            raise FileNotFoundError("No hay cache cargado.")
    
        ruta_fil = os.path.join(self.cache_ultimo, "filtrado", "eeg_filtrado.npy")
        ruta_t = os.path.join(self.cache_ultimo, "crudo", "tiempo.npy")
    
        if not os.path.exists(ruta_fil):
            raise FileNotFoundError(f"No existe:\n{ruta_fil}")
        if not os.path.exists(ruta_t):
            raise FileNotFoundError(f"No existe:\n{ruta_t}")
    
        datos = np.load(ruta_fil, mmap_mode="r")
        tiempo = np.load(ruta_t, mmap_mode="r")
    
        return datos, tiempo
    
    def _redibujar_modo_actual(self):
        if not self.cache_ultimo:
            return

        modo = self._normalizar_modo_grafica(self.var_modo.get())
        if modo != self.var_modo.get():
            self.var_modo.set(modo)
        self.actualizar_nav_canal()

        if modo == "Filtrado multicanal":
            if self._vista_actual == "multicanal" and self._lines_multicanal:
                self._actualizar_filtrado_multicanal()
            else:
                self._graficar_filtrado_multicanal()

        elif modo == "Canal filtrado":
            if self._vista_actual == "canal" and self._line_canal is not None:
                self._actualizar_canal_filtrado()
            else:
                idx = self.combo_canal.current()
                if idx < 0:
                    idx = 0
                canal = idx + 1
                nombre = self.nombres_canales[idx]
                self._graficar_canal_filtrado(idx, canal, nombre)

        elif modo == "FFT filtrada":
            idx = self.combo_canal.current()
            if idx < 0:
                idx = 0
            canal = idx + 1
            nombre = self.nombres_canales[idx]
            self._graficar_fft_filtrada(idx, canal, nombre)

        elif modo == "Bandas":
            if self._vista_actual == "bandas" and self._lines_bandas:
                self._actualizar_bandas_multi()
            else:
                idx = self.combo_canal.current()
                if idx < 0:
                    idx = 0
                canal = idx + 1
                nombre = self.nombres_canales[idx]
                self._graficar_bandas_multi(idx, canal, nombre)

        elif modo == "Espectrograma":
            idx = self.combo_canal.current()
            if idx < 0:
                idx = 0
            canal = idx + 1
            nombre = self.nombres_canales[idx]
            self._graficar_espectrograma(idx, canal, nombre)
                
    def _limpiar_controles_tiempo(self):
        if hasattr(self, "_sl_t0") and self._sl_t0 is not None:
            self._sl_t0 = None
    
        if hasattr(self, "_sl_dt") and self._sl_dt is not None:
            self._sl_dt = None
    
        if hasattr(self, "_ax_sl_t0") and self._ax_sl_t0 is not None:
            try:
                self._ax_sl_t0.remove()
            except Exception:
                pass
            self._ax_sl_t0 = None
    
        if hasattr(self, "_ax_sl_dt") and self._ax_sl_dt is not None:
            try:
                self._ax_sl_dt.remove()
            except Exception:
                pass
            self._ax_sl_dt = None

    ## =========================================================
    ## Interacción con mouse: arrastre y tooltips
    ## =========================================================
    def _desactivar_pan_tiempo(self):
        if self._cid_press is not None:
            self.canvas.mpl_disconnect(self._cid_press)
            self._cid_press = None
    
        if self._cid_motion is not None:
            self.canvas.mpl_disconnect(self._cid_motion)
            self._cid_motion = None
    
        if self._cid_release is not None:
            self.canvas.mpl_disconnect(self._cid_release)
            self._cid_release = None
    
        self._drag_tiempo_activo = False
        self._drag_x_inicio = None
        self._drag_t0_inicio = None
        self._drag_ultimo_t0 = None
        self._drag_px_por_seg = None
        
        

    
    def _activar_pan_tiempo(self):
        self._desactivar_pan_tiempo()

        def on_press(event):
            modo = self._normalizar_modo_grafica(self.var_modo.get())
            if modo != self.var_modo.get():
                self.var_modo.set(modo)

            if modo in ("FFT filtrada", "Espectrograma"):
                return

            if event.inaxes is None:
                return

            if self._ax_sl_dt is not None and event.inaxes == self._ax_sl_dt:
                return

            if hasattr(self, "toolbar") and getattr(self.toolbar, "mode", ""):
                return

            if event.button != 1:
                return

            if event.x is None:
                return

            ax = self.fig.axes[0] if self.fig.axes else None
            if ax is None:
                return

            # ancho visible del eje en píxeles
            try:
                x0_disp = ax.transData.transform((ax.get_xlim()[0], 0))[0]
                x1_disp = ax.transData.transform((ax.get_xlim()[1], 0))[0]
                ancho_px = abs(x1_disp - x0_disp)
            except Exception:
                return

            if ancho_px < 10:
                return

            dt_visible = max(1e-6, float(self._dt))
            self._drag_px_por_seg = ancho_px / dt_visible

            self._drag_tiempo_activo = True
            self._drag_x_inicio = float(event.x)     # en píxeles
            self._drag_t0_inicio = float(self._t0)
            self._drag_ultimo_t0 = float(self._t0)

        def on_motion(event):
            if not self._drag_tiempo_activo:
                return

            if event.x is None:
                return

            fs, tf_total = self._leer_fs_y_tf()
            if tf_total is None or tf_total <= 0:
                return

            if self._drag_px_por_seg is None or self._drag_px_por_seg <= 0:
                return

            dt = max(1.0, float(self._dt))

            # delta en píxeles, NO en xdata
            dx_px = float(event.x) - self._drag_x_inicio

            # convertir píxeles a segundos
            dx_seg = dx_px / self._drag_px_por_seg

            factor_arrastre = 1.0
            nuevo_t0 = self._drag_t0_inicio - dx_seg * factor_arrastre
            nuevo_t0 = max(0.0, min(nuevo_t0, max(0.0, tf_total - dt)))

            if self._drag_ultimo_t0 is not None and abs(nuevo_t0 - self._drag_ultimo_t0) < 0.02:
                return

            self._t0 = nuevo_t0
            self._drag_ultimo_t0 = nuevo_t0
            self._redibujar_modo_actual()

        def on_release(event):
            self._drag_tiempo_activo = False
            self._drag_x_inicio = None
            self._drag_t0_inicio = None
            self._drag_ultimo_t0 = None
            self._drag_px_por_seg = None

        self._cid_press = self.canvas.mpl_connect("button_press_event", on_press)
        self._cid_motion = self.canvas.mpl_connect("motion_notify_event", on_motion)
        self._cid_release = self.canvas.mpl_connect("button_release_event", on_release)
        
    def _activar_hover_multicanal(self):
        if self._cid_hover_multicanal is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_hover_multicanal)
            except Exception:
                pass
            self._cid_hover_multicanal = None
    
        def on_motion(event):
            if self._vista_actual != "multicanal":
                self._tooltip_ocultar()
                self._canal_hover_actual = None
                return
    
            if self._ax_main is None:
                self._tooltip_ocultar()
                self._canal_hover_actual = None
                return
    
            # fuera del canvas de matplotlib
            if event.x is None or event.y is None:
                self._tooltip_ocultar()
                self._canal_hover_actual = None
                return
    
            try:
                renderer = self.canvas.get_renderer()
            except Exception:
                self._tooltip_ocultar()
                self._canal_hover_actual = None
                return
    
            # -------------------------------------------------
            # 1) Hover exacto sobre texto del eje Y
            #    AQUÍ la clave es usar event.x / event.y
            # -------------------------------------------------
            labels = self._ax_main.get_yticklabels()
    
            for i, lab in enumerate(labels):
                if i >= len(self.nombres_canales):
                    continue
    
                try:
                    bbox = lab.get_window_extent(renderer=renderer)
                except Exception:
                    continue
    
                if bbox.contains(event.x, event.y):
                    nombre = self.nombres_canales[i]
                    self._canal_hover_actual = i
                    self._tooltip_mostrar(
                        self.raiz.winfo_pointerx(),
                        self.raiz.winfo_pointery(),
                        f"Canal: {nombre}"
                    )
                    return
    
            # -------------------------------------------------
            # 2) Si no cayó sobre el texto, usar cercanía a la fila
            # -------------------------------------------------
            if event.inaxes == self._ax_main and event.ydata is not None and self._offsets_multicanal is not None:
                offsets = np.asarray(self._offsets_multicanal, dtype=np.float64)
                if len(offsets) > 0:
                    y = float(event.ydata)
                    idx = int(np.argmin(np.abs(offsets - y)))
                    dist = abs(offsets[idx] - y)
    
                    if len(offsets) > 1:
                        sep = float(np.median(np.diff(offsets)))
                    else:
                        sep = 1.0
    
                    if dist <= 0.45 * sep:
                        nombre = self.nombres_canales[idx]
                        self._canal_hover_actual = idx
                        self._tooltip_mostrar(
                            self.raiz.winfo_pointerx(),
                            self.raiz.winfo_pointery(),
                            f"Canal: {nombre}"
                        )
                        return
    
            self._tooltip_ocultar()
            self._canal_hover_actual = None
    
        self._cid_hover_multicanal = self.canvas.mpl_connect("motion_notify_event", on_motion)
        
    def _desactivar_hover_multicanal(self):
        if self._cid_hover_multicanal is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_hover_multicanal)
            except Exception:
                pass
            self._cid_hover_multicanal = None
    
        self._canal_hover_actual = None
        self._tooltip_ocultar()
        
        
        
    def _mostrar_placeholder_visor(self, texto="Selecciona y procesa un archivo para visualizar."):
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=12)
            self.canvas.draw_idle()
        except Exception:
            pass

    ## =========================================================
    ## Gráficas: multicanal, canal, FFT y bandas
    ## =========================================================
    def _graficar_filtrado_multicanal(self):
        try:
            datos, tiempo = self._cargar_filtrado_y_tiempo()
        except Exception as e:
            messagebox.showerror("Error", f"No pude cargar señal filtrada:\n{e}")
            return

        fs, tf_total = self._leer_fs_y_tf()
        fs, tf_total, tiempo = self._asegurar_fs_tiempo(datos, tiempo, fs, tf_total)
        self._fs = fs
        self._tf_total = tf_total

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
    
        dt = max(5.0, min(30.0, dt))
        self._dt = dt

        t1 = min(t0 + dt, tf_total if tf_total > 0 else float(datos.shape[1] / fs))
        i0 = max(0, int(round(t0 * fs)))
        i1 = min(datos.shape[1], int(round(t1 * fs)))

        if i1 <= i0:
            messagebox.showwarning("Ventana", "La ventana de tiempo no es válida.")
            return

        x = np.asarray(tiempo[i0:i1], dtype=np.float64)
        seg = np.asarray(datos[:, i0:i1], dtype=np.float64)

        if seg.size == 0:
            messagebox.showwarning("Datos", "No hay datos en la ventana seleccionada.")
            return

        amp_ref = np.percentile(np.abs(seg), 95)
        if not np.isfinite(amp_ref) or amp_ref <= 1e-12:
            amp_ref = 1.0

       # ganancia = float(self.var_ganancia.get())

        try:
            ganancia = float(self.var_ganancia.get())
        except Exception:
            ganancia = 1.0
            
        
    
        separacion = 4.0 * amp_ref * max(1.0, ganancia)
        offsets = np.arange(seg.shape[0], dtype=np.float64) * separacion

        self.fig.clear()
        self._ax_main = self.fig.add_subplot(111)
        self._lines_multicanal = []
        self._offsets_multicanal = offsets

        for ch in range(seg.shape[0]):
            y = seg[ch] * ganancia + offsets[ch]
            (linea,) = self._ax_main.plot(x, y, linewidth=0.7)
            self._lines_multicanal.append(linea)

        self._ax_main.set_title("Señal EEG Multicanal")
        self._ax_main.set_xlabel("Tiempo (s)")
        self._ax_main.set_ylabel("Canales")
        self._ax_main.set_yticks(offsets)
        self._ax_main.set_yticklabels(self.nombres_canales, fontsize=6)
        self._ax_main.invert_yaxis()
        self._ax_main.grid(True, alpha=0.25)
        self._ax_main.set_xlim(float(x[0]), float(x[-1]))

        self._vista_actual = "multicanal"
        self._line_canal = None
        self._axes_bandas = []
        self._lines_bandas = []

        self._limpiar_controles_tiempo()
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.07)
        self.canvas.draw_idle()
        self._activar_hover_multicanal()
        self._actualizar_barra_modo_grafica("Filtrado multicanal")
        
    def _actualizar_filtrado_multicanal(self):
        if not self._lines_multicanal or self._ax_main is None:
            return

        try:
            datos, tiempo = self._cargar_filtrado_y_tiempo()
        except Exception:
            return

        fs, tf_total = self._leer_fs_y_tf()
        fs, tf_total, tiempo = self._asegurar_fs_tiempo(datos, tiempo, fs, tf_total)
        self._fs = fs
        self._tf_total = tf_total

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
            
        dt = max(5.0, min(30.0, dt))
        self._dt = dt

        t1 = min(t0 + dt, tf_total if tf_total > 0 else float(datos.shape[1] / fs))
        i0 = max(0, int(round(t0 * fs)))
        i1 = min(datos.shape[1], int(round(t1 * fs)))

        if i1 <= i0:
            return

        x = np.asarray(tiempo[i0:i1], dtype=np.float64)
        seg = np.asarray(datos[:, i0:i1], dtype=np.float64)

        if seg.size == 0:
            return

        amp_ref = np.percentile(np.abs(seg), 95)
        if not np.isfinite(amp_ref) or amp_ref <= 1e-12:
            amp_ref = 1.0

        #ganancia = float(self.var_ganancia.get())
        try:
            ganancia = float(self.var_ganancia.get())
        except Exception:
            ganancia = 1.0
            
        separacion = 4.0 * amp_ref * max(1.0, ganancia)
        offsets = np.arange(seg.shape[0], dtype=np.float64) * separacion
        self._offsets_multicanal = offsets

        if len(self._lines_multicanal) != seg.shape[0]:
            self._graficar_filtrado_multicanal()
            return

        for ch, linea in enumerate(self._lines_multicanal):
            y = seg[ch] * ganancia + offsets[ch]
            linea.set_data(x, y)

        self._ax_main.set_yticks(offsets)
        self._ax_main.set_yticklabels(self.nombres_canales, fontsize=6)
        self._ax_main.set_xlim(float(x[0]), float(x[-1]))

        ymin = offsets[0] - separacion
        ymax = offsets[-1] + separacion
        self._ax_main.set_ylim(ymax, ymin)

        self.canvas.draw_idle()
        #self._activar_hover_multicanal()
        
        
    def _graficar_canal_filtrado(self, idx, canal, nombre):
        try:
            datos, tiempo = self._cargar_filtrado_y_tiempo()
        except Exception as e:
            messagebox.showerror("Error", f"No pude cargar señal filtrada:\n{e}")
            return

        if idx < 0 or idx >= datos.shape[0]:
            messagebox.showwarning("Canal", "Índice de canal inválido.")
            return

        fs, tf_total = self._leer_fs_y_tf()
        fs, tf_total, tiempo = self._asegurar_fs_tiempo(datos, tiempo, fs, tf_total)
        self._fs = fs
        self._tf_total = tf_total

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
    
        dt = max(5.0, min(30.0, dt))
        self._dt = dt

        t1 = min(t0 + dt, tf_total if tf_total else float(datos.shape[1] / fs))
        i0 = max(0, int(round(t0 * fs)))
        i1 = min(datos.shape[1], int(round(t1 * fs)))

        if i1 <= i0:
            messagebox.showwarning("Ventana", "La ventana de tiempo no es válida.")
            return

        x = np.asarray(tiempo[i0:i1], dtype=np.float64)
        y = np.asarray(datos[idx, i0:i1], dtype=np.float64)

        self._idx_actual = idx
        self._canal_actual = canal
        self._nombre_actual = nombre

        self.fig.clear()
        self._ax_main = self.fig.add_subplot(111)
        (self._line_canal,) = self._ax_main.plot(x, y, linewidth=0.9)

        self._ax_main.set_title(f"Canal filtrado: {nombre} (#{canal})")
        self._ax_main.set_xlabel("Tiempo (s)")
        self._ax_main.set_ylabel("μV")
        self._ax_main.grid(True, alpha=0.3)

        if len(x) > 1:
            self._ax_main.set_xlim(float(x[0]), float(x[-1]))

        self._vista_actual = "canal"
        self._lines_multicanal = []
        self._axes_bandas = []
        self._lines_bandas = []

        self._limpiar_controles_tiempo()
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._desactivar_hover_multicanal()
        self._actualizar_barra_modo_grafica("Canal filtrado", nombre)


    def _actualizar_canal_filtrado(self):
        if self._line_canal is None or self._ax_main is None:
            return

        try:
            datos, tiempo = self._cargar_filtrado_y_tiempo()
        except Exception:
            return

        idx = self._idx_actual
        if idx is None or idx < 0 or idx >= datos.shape[0]:
            return

        fs, tf_total = self._leer_fs_y_tf()
        fs, tf_total, tiempo = self._asegurar_fs_tiempo(datos, tiempo, fs, tf_total)
        self._fs = fs
        self._tf_total = tf_total

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
        dt = max(5.0, min(30.0, dt))
        self._dt = dt

        t1 = min(t0 + dt, tf_total if tf_total else float(datos.shape[1] / fs))
        i0 = max(0, int(round(t0 * fs)))
        i1 = min(datos.shape[1], int(round(t1 * fs)))

        if i1 <= i0:
            return

        x = np.asarray(tiempo[i0:i1], dtype=np.float64)
        y = np.asarray(datos[idx, i0:i1], dtype=np.float64)

        self._line_canal.set_data(x, y)
        if len(x) > 1:
            self._ax_main.set_xlim(float(x[0]), float(x[-1]))

        self._ax_main.relim()
        self._ax_main.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()

    def _activar_grafica_inicial(self):
        if not self.cache_ultimo:
            self._mostrar_marco_visor_grafica(False)
            self._mostrar_tabs_grafica(False)
            self._mostrar_placeholder_visor("Sin datos cargados.")
            return

        modo_actual = self._normalizar_modo_grafica(self.var_modo.get())
        self._mostrar_marco_visor_grafica(True)
        self._mostrar_tabs_grafica(True)
        if modo_actual == "FFT filtrada":
            self.var_modo.set("FFT filtrada")
        else:
            self.var_modo.set("Filtrado multicanal")
        self._refrescar_ui_modo()
    
        try:
            self.combo_canal.current(0)
            self.var_canal_nombre.set(self.nombres_canales[0])
        except Exception:
            pass
    
        self._t0 = 0.0
        self.hay_grafico_activo = False
        self.raiz.after(80, self.graficar)
       
    def _graficar_fft_filtrada(self, idx, canal, nombre_canal):
        if not self.cache_ultimo:
            return

        ruta_f = os.path.join(self.cache_ultimo, "fft", "freqs_filtrado.npy")
        ruta_a = os.path.join(self.cache_ultimo, "fft", "fft_amp_filtrado.npy")

        try:
            freqs = np.load(ruta_f, mmap_mode="r")
        except Exception as e:
            messagebox.showerror("Error", f"No pude cargar la FFT filtrada:\n{e}")
            return

        try:
            amps = np.load(ruta_a, mmap_mode="r")
        except Exception as e:
            messagebox.showerror("Error", f"No pude cargar la FFT filtrada:\n{e}")
            return

        self.log("[OK] FFT cargada desde cache actualizado.")

        if idx < 0 or idx >= amps.shape[0]:
            messagebox.showwarning("Canal", "Índice de canal inválido para FFT.")
            return

        try:
            fmax = float(self.var_fmax.get())
        except Exception:
            fmax = 40.0
        mask = freqs <= fmax

        x = np.asarray(freqs[mask], dtype=np.float64)
        y = np.asarray(amps[idx, mask], dtype=np.float64)
        unidad_fft = self._leer_unidad_fft_desde_cache()
        if unidad_fft == "µV":
            ylabel = " (µV)"
        elif unidad_fft:
            ylabel = f" ({unidad_fft})"
        else:
            ylabel = "(u.a.)"

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        bandas_fft = [
            ("Delta", 1, 3,  "#7EC8E3"),
            ("Theta", 3, 8,  "#8BC34A"),
            ("Alfa",  8, 12, "#FFD54F"),
            ("Beta", 12, 30, "#FF8A65")
            
        ]

        # fondos por bandas
        bandas_color = [
            ("Delta", 1, 3, "#dbeafe"),
            ("Theta", 3, 8, "#dcfce7"),
            ("Alfa", 8, 12, "#fef3c7"),
            ("Beta", 12, 30, "#fee2e2"),
        ]

        for _, f0, f1, col in bandas_color:
            if f0 < fmax:
                ax.axvspan(f0, min(f1, fmax), alpha=0.35, color=col)

        #fmax = float(self.var_fmax.get())
        
        for nombre_banda, f0, f1, color in bandas_fft:
            if f0 >= fmax:
                continue
            ax.axvspan(f0, min(f1, fmax), alpha=0.12, color=color)

        # fft_amp_filtrado.npy ya viene normalizada por N en el pipeline.
        # La unidad se conserva si la señal original estaba en microvoltios.
        ax.plot(x, y, linewidth=1.0)
        
        # Leyenda de bandas visibles

        handles = []
        for nombre_banda, f0, f1, col in bandas_color:
            if f0 < fmax:
                etiqueta = f"{nombre_banda} ({f0}-{min(f1, fmax):g} Hz)"
                handles.append(
                    Patch(
                        facecolor=col,
                        edgecolor="gray",
                        label=etiqueta,
                        alpha=0.8
                    )
                )
        
        if handles:
            ax.legend(
                handles=handles,
                loc="upper right",
                title="Bandas",
                fontsize=8,
                title_fontsize=9,
                frameon=True
            )

        ax.set_title(f"FFT filtrada - Canal {nombre_canal} (#{canal})")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if len(x) > 1:
            ax.set_xlim(0, fmax)
        
        self._vista_actual = "fft"
        self._line_canal = None
        self._lines_multicanal = []
        self._axes_bandas = []
        self._lines_bandas = []
            
        self._limpiar_controles_tiempo()
        self._desactivar_pan_tiempo()
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._desactivar_hover_multicanal()
        self._actualizar_barra_modo_grafica("FFT filtrada", nombre_canal)

    def _graficar_espectrograma(self, idx, canal, nombre_canal):
        if not self.cache_ultimo:
            return

        ruta_fil = os.path.join(self.cache_ultimo, "filtrado", "eeg_filtrado.npy")
        if not os.path.exists(ruta_fil):
            msg = "No se encontró la señal filtrada. Procese el archivo antes de visualizar el espectrograma."
            self.log(msg)
            messagebox.showwarning("Espectrograma", msg)
            return

        try:
            datos = np.load(ruta_fil, mmap_mode="r")
        except Exception as e:
            messagebox.showerror("Espectrograma", f"No pude cargar la señal filtrada:\n{e}")
            return

        if idx < 0 or idx >= datos.shape[0]:
            messagebox.showwarning("Canal", "El canal seleccionado no existe en la señal filtrada.")
            return

        fs = self._leer_fs_desde_meta_cache(fs_defecto=500.0)
        x = np.asarray(datos[idx], dtype=np.float64)
        if x.size < 2:
            messagebox.showwarning("Espectrograma", "No hay muestras suficientes para calcular el espectrograma.")
            return

        ventana_seg = 2.0
        solapamiento = 0.50
        solapamiento_pct = 50.0

        try:
            fmax = float(self.var_fmax.get())
        except Exception:
            fmax = 40.0
        fmax = max(1.0, min(40.0, fmax))

        nperseg = int(round(ventana_seg * fs))
        nperseg = max(2, nperseg)
        if nperseg > x.size:
            nperseg = int(x.size)
            self.log("La ventana del espectrograma era mayor que la señal. Se ajustó automáticamente.")

        noverlap = int(round(nperseg * solapamiento))
        if noverlap >= nperseg:
            noverlap = max(0, nperseg - 1)

        self.log(f"Espectrograma por canal: {nombre_canal} | ventana=2 s | solapamiento=50 % | escala=dB")

        f, t, sxx = spectrogram(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd"
        )

        mask = (f >= 1.0) & (f <= fmax)
        f = np.asarray(f[mask], dtype=np.float64)
        sxx = np.asarray(sxx[mask, :], dtype=np.float64)

        if f.size == 0 or sxx.size == 0:
            messagebox.showwarning("Espectrograma", "No hay contenido espectral en el rango seleccionado.")
            return

        sxx_db = 10.0 * np.log10(sxx + 1e-12)

        self.fig.clear()
        gs = self.fig.add_gridspec(
            1, 3,
            width_ratios=[1.0, 0.08, 0.035],
            left=0.07,
            right=0.88,
            bottom=0.11,
            top=0.90,
            wspace=0.06
        )
        ax = self.fig.add_subplot(gs[0, 0])

        # No usar sharey=ax, porque al ocultar ticks en ax_bandas
        # también se pueden ocultar los ticks del eje principal.
        ax_bandas = self.fig.add_subplot(gs[0, 1])

        cax = self.fig.add_subplot(gs[0, 2])

        mesh = ax.pcolormesh(t, f, sxx_db, shading="auto", cmap="viridis")

        for yref in (4.0, 8.0, 12.0, 30.0):
            if yref <= fmax:
                ax.axhline(yref, color="#F3F4F6", linewidth=0.8, alpha=0.75, zorder=3)

        colorbar = self.fig.colorbar(mesh, cax=cax)
        colorbar.set_label("Potencia espectral (dB)")
        colorbar.outline.set_linewidth(0.6)

        ax.set_title(f"Espectrograma EEG - Canal {nombre_canal}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_ylim(1.0, fmax)

        ticks_y = [1, 4, 8, 12, 20, 30, 40]
        ticks_y = [v for v in ticks_y if v <= fmax]

        ax.set_yticks(ticks_y)
        ax.set_yticklabels([str(v) for v in ticks_y])
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            labelleft=True,
            labelsize=10,
            colors="black"
        )
        ax.grid(False)

        ax_bandas.set_xlim(0.0, 1.0)
        ax_bandas.set_ylim(1.0, fmax)
        ax_bandas.set_xticks([])
        ax_bandas.set_yticks([])

        for spine in ax_bandas.spines.values():
            spine.set_visible(False)

        ax_bandas.patch.set_alpha(0.0)

        bandas_texto = [
            ("Delta", 1.0, 4.0),
            ("Theta", 4.0, 8.0),
            ("Alfa", 8.0, 12.0),
            ("Beta", 12.0, 30.0),
        ]
        for nombre_banda, f0, f1 in bandas_texto:
            if f0 >= fmax:
                continue
            y0 = max(1.0, f0)
            y1 = min(f1, fmax)
            if y1 <= y0:
                continue
            y_centro = 0.5 * (y0 + y1)
            ax_bandas.text(
                0.02,
                y_centro,
                nombre_banda,
                ha="left",
                va="center",
                fontsize=9,
                color="#4B5563"
            )

        self._vista_actual = "espectrograma"
        self._line_canal = None
        self._lines_multicanal = []
        self._axes_bandas = []
        self._lines_bandas = []
        self._idx_actual = idx
        self._canal_actual = canal
        self._nombre_actual = nombre_canal
        self._ax_main = ax

        self._limpiar_controles_tiempo()
        self._desactivar_pan_tiempo()
        self._desactivar_hover_multicanal()
        self.canvas.draw_idle()
        self.raiz.after_idle(self._dibujar_mapa_regional_resumen)
        self._actualizar_barra_modo_grafica("Espectrograma", nombre_canal)

    def graficar(self):
        self.ir_a_visor()

        if not self.cache_ultimo:
            messagebox.showwarning("Sin cache", "Primero procesa un archivo para poder graficar.")
            return

        idx = self.combo_canal.current()
        if idx < 0:
            idx = 0

        canal = idx + 1
        nombre = self.nombres_canales[idx]
        modo = self._normalizar_modo_grafica(self.var_modo.get())
        if modo != self.var_modo.get():
            self.var_modo.set(modo)
            self._refrescar_ui_modo()

        self.canal_activo_simple = nombre.upper().replace("-AVG", "")
        self.hay_grafico_activo = True
        self.actualizar_nav_canal()

        # Mostrar GIF mientras matplotlib renderiza
        if hasattr(self, "_loading") and not getattr(self, "_resizing", False):
            self._loading.show()
            self.raiz.update_idletasks()

        if modo not in ("FFT filtrada", "Espectrograma"):
            if self._cid_press is None or self._cid_motion is None or self._cid_release is None:
                self._activar_pan_tiempo()
        if modo == "Filtrado multicanal":
            self._graficar_filtrado_multicanal()
        elif modo == "Canal filtrado":
            self._graficar_canal_filtrado(idx, canal, nombre)
        elif modo == "FFT filtrada":
            self._graficar_fft_filtrada(idx, canal, nombre)
        elif modo == "Bandas":
            self._graficar_bandas_multi(idx, canal, nombre)
        elif modo == "Espectrograma":
            self._graficar_espectrograma(idx, canal, nombre)
        else:
            messagebox.showwarning("Modo", f"Modo no reconocido: {modo}")

        # Ocultar GIF tras el render
        if hasattr(self, "_loading"):
            self.raiz.after(120, self._loading.hide)

    ## =========================================================
    ## Bandas: búsqueda robusta en cache
    ## =========================================================
    def _normalizar_nombre(self, s: str) -> str:
        s = s.lower().strip()
        # normaliza alfa/alpha por si el pipeline usa inglés
        s = s.replace("alpha", "alfa")
        # quita tildes por si acaso
        s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        return s

    def _buscar_archivo_banda(self, banda: str):
        """
        Busca el archivo de la banda en TODO el cache.
        Acepta .npy y .npz y nombres tipo Alfa/Alpha.
        """
        if not self.cache_ultimo:
            return None

        bnorm = self._normalizar_nombre(banda)

        candidatos_carpetas = [
            os.path.join(self.cache_ultimo, "bandas"),
            os.path.join(self.cache_ultimo, "Bandas"),
            os.path.join(self.cache_ultimo, "bands"),
            os.path.join(self.cache_ultimo, "Bands"),
        ]

        def probar_en(carpeta):
            if not os.path.isdir(carpeta):
                return None
            for nombre in os.listdir(carpeta):
                base, ext = os.path.splitext(nombre)
                if ext.lower() not in (".npy", ".npz"):
                    continue
                n = self._normalizar_nombre(base)
                if n == bnorm:
                    return os.path.join(carpeta, nombre)
            return None

        for c in candidatos_carpetas:
            r = probar_en(c)
            if r:
                return r

        for root, _, files in os.walk(self.cache_ultimo):
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() not in (".npy", ".npz"):
                    continue
                n = self._normalizar_nombre(base)
                if n == bnorm:
                    return os.path.join(root, f)

        return None

    def _cargar_matriz_banda(self, ruta):
        """
        Devuelve matriz (canales x muestras).
        Soporta .npy y .npz.
        """
        if ruta.lower().endswith(".npz"):
            z = np.load(ruta, allow_pickle=True)
            key = list(z.keys())[0]
            return z[key]
        else:
            return np.load(ruta, mmap_mode="r", allow_pickle=True)

    def _graficar_bandas_multi(self, idx, canal, nombre_canal):
        seleccion = [b for b in self.bandas if self.band_vars[b].get()]
        if not seleccion:
            seleccion = ["Alfa"]

        fs, tf_total = self._leer_fs_y_tf()
        if fs is None:
            messagebox.showerror("fs", "No pude obtener fs (falta fs.npy o no se puede inferir con tiempo.npy).")
            return

        ruta_b0 = self._buscar_archivo_banda(seleccion[0])
        if not ruta_b0 or not os.path.exists(ruta_b0):
            self.log("No pude encontrar archivo de bandas en el cache.")
            messagebox.showerror("Bandas", "No encontré el archivo de la banda.")
            return

        datos0 = self._cargar_matriz_banda(ruta_b0)
        N = datos0.shape[1]
        tf_total = float(N / fs)

        self._fs = float(fs)
        self._tf_total = tf_total

        self._idx_actual = idx
        self._canal_actual = canal
        self._nombre_actual = nombre_canal
        self._bandas_actuales = seleccion

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
        dt = max(5.0, min(30.0, dt))
        t0 = min(t0, max(0.0, tf_total - dt))
        self._t0 = t0
        self._dt = dt

        self.fig.clear()
        n = len(seleccion)
        self._axes_bandas = [self.fig.add_subplot(n, 1, i + 1) for i in range(n)]
        self._lines_bandas = []

        self.fig.suptitle(f"{nombre_canal} (#{canal})", fontsize=11, y=0.98)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.16, hspace=0.90)

        for i, b in enumerate(seleccion):
            ax = self._axes_bandas[i]
            ruta_b = self._buscar_archivo_banda(b)
            if not ruta_b or not os.path.exists(ruta_b):
                ax.set_title(f"{b} (NO ENCONTRADO)", fontsize=10)
                ax.grid(alpha=0.3)
                self._lines_bandas.append(None)
                continue

            datos = self._cargar_matriz_banda(ruta_b)
            if idx < 0 or idx >= datos.shape[0]:
                ax.set_title(f"{b} | Canal inválido", fontsize=10)
                ax.grid(alpha=0.3)
                self._lines_bandas.append(None)
                continue

            i0 = int(t0 * self._fs)
            i1 = i0 + int(dt * self._fs)
            i0 = max(i0, 0)
            i1 = min(i1, datos.shape[1])

            x = np.arange(i0, i1) / self._fs
            y = datos[idx, i0:i1]

            (linea,) = ax.plot(x, y, linewidth=0.9)
            self._lines_bandas.append(linea)

            ax.set_title(f"{b}", fontsize=10)
            if i == n - 1:
                ax.set_xlabel("Tiempo (s)", labelpad=8)
            else:
                ax.set_xlabel("")
            ax.set_ylabel("μV")
            ax.grid(alpha=0.3)

        self._vista_actual = "bandas"
        self._line_canal = None
        self._lines_multicanal = []

        self._limpiar_controles_tiempo()
        self.canvas.draw_idle()
        self._desactivar_hover_multicanal()

        self.log(f"Bandas multi: {', '.join(seleccion)} | {nombre_canal} (#{canal})")
        self._actualizar_barra_modo_grafica("Bandas", nombre_canal)

    def _actualizar_bandas_multi(self):
        if not self._axes_bandas or not self._lines_bandas or not self._bandas_actuales:
            return

        if self._fs is None or self._fs <= 0:
            return

        idx = self._idx_actual
        if idx is None:
            return

        t0 = max(0.0, float(self._t0))
        try:
            dt = float(self.var_ventana.get())
        except Exception:
            dt = 5.0
        dt = max(5.0, min(30.0, dt))
        self._dt = dt

        if self._tf_total is not None:
            t0 = min(t0, max(0.0, self._tf_total - dt))

        self._t0 = t0
        self._dt = dt

        for i, b in enumerate(self._bandas_actuales):
            ax = self._axes_bandas[i]
            linea = self._lines_bandas[i]

            if linea is None:
                continue

            ruta_b = self._buscar_archivo_banda(b)
            if not ruta_b or not os.path.exists(ruta_b):
                continue

            datos = self._cargar_matriz_banda(ruta_b)
            if idx < 0 or idx >= datos.shape[0]:
                continue

            i0 = int(t0 * self._fs)
            i1 = i0 + int(dt * self._fs)
            i0 = max(i0, 0)
            i1 = min(i1, datos.shape[1])

            x = np.arange(i0, i1) / self._fs
            y = datos[idx, i0:i1]

            linea.set_data(x, y)
            if len(x) > 1:
                ax.set_xlim(float(x[0]), float(x[-1]))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        self.canvas.draw_idle()
        

## Punto de entrada
if __name__ == "__main__":
    habilitar_alta_resolucion_windows()
    raiz = tk.Tk()
    aplicar_escala_tk(raiz)
    app = AppEEG(raiz)
    raiz.mainloop()