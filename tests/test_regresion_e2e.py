# -*- coding: utf-8 -*-
"""
T-5 — Regresión end-to-end del pipeline NeuroX.

Pasa un registro EEG sintético reproducible por el flujo completo
(procesar_archivo -> CACHE -> generar_informe_desde_cache -> PDF) en una
carpeta temporal y compara la "huella" de resultados contra
tests/referencias/e2e_sintetico.json. Opcionalmente repite la comparación
con los 5 registros reales de la auditoría si se define la variable de
entorno NEUROX_DATOS_REGRESION con la carpeta de los .dat/.dap.

No toca la CACHE ni los informes del usuario.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -p "test_*.py" -v
Saltar las pruebas lentas (~1 min):        NEUROX_SALTAR_E2E=1
Incluir registros reales (~3 min más):     NEUROX_DATOS_REGRESION=C:/ruta/datosTodos
Regenerar referencias tras un cambio intencional:
    python tests/regresion_e2e.py --actualizar [--reales C:/ruta/datosTodos]
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

TESTS = os.path.dirname(os.path.abspath(__file__))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import regresion_e2e as R  # noqa: E402

SALTAR_E2E = os.environ.get("NEUROX_SALTAR_E2E", "").strip() not in ("", "0")
CARPETA_REALES = os.environ.get(R.ENV_DATOS_REALES, "").strip()


def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _cargar_referencia(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


class TestComparadorHuellas(unittest.TestCase):
    """Las tolerancias deben absorber ruido numérico, no cambios clínicos."""

    def test_huellas_identicas_no_tienen_diferencias(self):
        h = {"bandas.global.Alfa": 25.0, "semaforo.Theta_Alfa": "Normal"}
        self.assertEqual(R.comparar_huellas(h, dict(h)), [])

    def test_variacion_dentro_de_tolerancia_absoluta(self):
        ref = {"bandas.global.Alfa": 25.0}
        self.assertEqual(R.comparar_huellas(ref, {"bandas.global.Alfa": 25.4}), [])

    def test_variacion_fuera_de_tolerancia_absoluta(self):
        ref = {"bandas.global.Alfa": 25.0}
        difs = R.comparar_huellas(ref, {"bandas.global.Alfa": 25.6})
        self.assertEqual(len(difs), 1)
        self.assertIn("bandas.global.Alfa", difs[0])

    def test_tolerancia_relativa_en_ratios(self):
        ref = {"ratios.global.Delta_Alfa": 1.00}
        self.assertEqual(R.comparar_huellas(ref, {"ratios.global.Delta_Alfa": 1.029}), [])
        self.assertEqual(len(R.comparar_huellas(ref, {"ratios.global.Delta_Alfa": 1.031})), 1)

    def test_categorias_y_canales_se_comparan_exactos(self):
        ref = {"semaforo.Theta_Alfa": "Normal", "calidad.canales_atipicos": ["PO7", "T7"]}
        act = {"semaforo.Theta_Alfa": "Borderline", "calidad.canales_atipicos": ["T7"]}
        self.assertEqual(len(R.comparar_huellas(ref, act)), 2)

    def test_none_contra_numero_es_diferencia(self):
        self.assertEqual(len(R.comparar_huellas({"paf.global_hz": None}, {"paf.global_hz": 9.8})), 1)

    def test_claves_faltantes_y_nuevas_se_reportan(self):
        difs = R.comparar_huellas({"a": 1}, {"b": 1})
        self.assertEqual(len(difs), 2)


class TestRegistroSintetico(unittest.TestCase):
    """El generador debe ser determinista y legible por el lector real."""

    def test_misma_semilla_mismos_bytes_y_otra_semilla_distintos(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            ra = R.generar_registro_sintetico(a, duracion_s=10.0)
            rb = R.generar_registro_sintetico(b, duracion_s=10.0)
            rc = R.generar_registro_sintetico(c, semilla=8, duracion_s=10.0)
            self.assertEqual(_sha256(ra), _sha256(rb))
            self.assertNotEqual(_sha256(ra), _sha256(rc))

    def test_lector_de_neurox_acepta_el_registro(self):
        import pipeline_eeg_completo as P
        with tempfile.TemporaryDirectory() as tmp:
            ruta = R.generar_registro_sintetico(tmp, duracion_s=10.0)
            datos, _t, fs, meta = P.cargar_datos_eeg(ruta, logger=lambda m: None)
            self.assertEqual(datos.shape, (64, 5000))
            self.assertEqual(fs, 500.0)
            self.assertEqual(meta["num_channels_total"], 69)
            del datos


@unittest.skipIf(SALTAR_E2E, "NEUROX_SALTAR_E2E activo")
class TestPipelineEndToEnd(unittest.TestCase):
    """Registro sintético por el flujo completo, una sola vez para la clase."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.tmp = tempfile.mkdtemp(prefix="neurox_e2e_")
        ruta = R.generar_registro_sintetico(cls.tmp)
        cls.cache, cls.pdf = R.procesar_en_temporal(ruta, cls.tmp, con_informe=True)
        cls.huella = R.extraer_huella(cls.cache)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_todo_queda_dentro_de_la_carpeta_temporal(self):
        tmp = os.path.normcase(os.path.realpath(self.tmp))
        for ruta in (self.cache, self.pdf):
            self.assertTrue(os.path.normcase(os.path.realpath(ruta)).startswith(tmp), ruta)

    def test_estructura_de_cache(self):
        for rel in (
            "meta.txt",
            os.path.join("analisis_welch", "pot_rel_promedios.json"),
            os.path.join("analisis_welch", "ratios_promedios_principales.json"),
            os.path.join("calidad_senal", "resumen_calidad.json"),
            os.path.join("canales_atipicos", "reporte_canales_atipicos.txt"),
            os.path.join("ica", "calidad_ica.json"),
        ):
            self.assertTrue(os.path.exists(os.path.join(self.cache, rel)), rel)

    def test_huella_igual_a_la_referencia(self):
        if not os.path.exists(R.REF_SINTETICO):
            self.skipTest("Falta la referencia; ejecutar: python tests/regresion_e2e.py --actualizar")
        difs = R.comparar_huellas(_cargar_referencia(R.REF_SINTETICO), self.huella)
        self.assertEqual(difs, [], "Cambios respecto a la referencia:\n  " + "\n  ".join(difs))

    # --- Comprobaciones fisiológicas independientes de la referencia ---
    def test_detecta_los_canales_defectuosos_inyectados(self):
        atipicos = set(self.huella["calidad.canales_atipicos"])
        afectados = atipicos | set(self.huella["calidad.canales_sospechosos"])
        self.assertIn("T7", atipicos)        # ruido de alta frecuencia
        self.assertIn("PO7", atipicos)       # canal casi plano
        self.assertIn("C4", afectados)       # electrodo suelto

    def test_paf_en_el_alfa_inyectado_de_10_hz(self):
        self.assertAlmostEqual(self.huella["paf.global_hz"], 10.0, delta=0.5)

    def test_alfa_mayor_atras_que_adelante(self):
        self.assertGreater(self.huella["alfa.occipital"], self.huella["alfa.frontal"])
        self.assertGreater(self.huella["gradiente_ap"], 1.0)

    def test_bandas_relativas_suman_cerca_de_100(self):
        total = sum(self.huella[f"bandas.global.{b}"] for b in ("Delta", "Theta", "Alfa", "Beta"))
        self.assertGreater(total, 85.0)
        self.assertLessEqual(total, 100.0 + 1e-6)

    def test_mide_la_red_de_60_hz_inyectada(self):
        self.assertGreaterEqual(self.huella["calidad.pct_ruido_red"], 30.0)
        self.assertTrue(self.huella["adquisicion.interferencia_red"])
        self.assertGreater(self.huella["calidad.snr_sin_red_db"], self.huella["calidad.snr_global_db"])

    def test_pdf_generado_con_las_secciones_principales(self):
        from pypdf import PdfReader
        self.assertTrue(os.path.exists(self.pdf))
        lector = PdfReader(self.pdf)
        self.assertGreaterEqual(len(lector.pages), 4)
        texto = " ".join(p.extract_text() or "" for p in lector.pages)
        for seccion in ("Informaci", "Calidad t", "Recomendaciones de adquisici",
                        "Potencia relativa por bandas", "Proporciones de bandas", "Resumen descriptivo"):
            self.assertIn(seccion, texto, seccion)


@unittest.skipIf(SALTAR_E2E, "NEUROX_SALTAR_E2E activo")
@unittest.skipUnless(CARPETA_REALES, f"Definir {R.ENV_DATOS_REALES} para incluir registros reales")
class TestRegresionRegistrosReales(unittest.TestCase):
    """Los 5 registros de la auditoría contra sus huellas de referencia."""

    def test_registros_reales_iguales_a_la_referencia(self):
        if not os.path.exists(R.REF_REALES):
            self.skipTest("Falta la referencia; ejecutar: python tests/regresion_e2e.py --actualizar --reales <carpeta>")
        warnings.filterwarnings("ignore")
        referencias = _cargar_referencia(R.REF_REALES)
        for nombre, ref in sorted(referencias.items()):
            origen = os.path.join(CARPETA_REALES, nombre)
            with self.subTest(registro=nombre):
                if not os.path.exists(origen):
                    self.skipTest(f"No existe {origen}")
                with tempfile.TemporaryDirectory(prefix="neurox_e2e_") as tmp:
                    cache = R.procesar_en_temporal(origen, tmp)
                    difs = R.comparar_huellas(ref, R.extraer_huella(cache))
                self.assertEqual(difs, [], f"{nombre}:\n  " + "\n  ".join(difs))


if __name__ == "__main__":
    unittest.main()
