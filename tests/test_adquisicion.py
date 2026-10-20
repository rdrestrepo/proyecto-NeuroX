# -*- coding: utf-8 -*-
"""
T-1 — Pruebas del diagnostico de adquisicion (Mejora 1).

Cubre las tres piezas que traducen el detector de canales atipicos a
recomendaciones de protocolo:

- _causa_principal_canal()
- detalle_atipicos_desde_reporte()
- diagnosticar_adquisicion()
- recomendaciones_adquisicion_desde_cache()

No procesa senales ni toca CACHE: usa caches temporales y detalles
sinteticos con la misma forma que produce detectar_canales_atipicos().

Ejecutar desde la raiz del proyecto:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import pipeline_eeg_completo as P  # noqa: E402


NOMBRES = P.obtener_nombres_canales_64()
N = len(NOMBRES)
CLAVES_Z = ("z_std", "z_ptp", "z_hf", "z_alpha", "z_dist")


def _detalle_vacio(n=N):
    """Detalle con la misma forma minima que consume el diagnostico."""
    detalle = {k: np.zeros(n, dtype=np.float64) for k in CLAVES_Z}
    detalle["score"] = np.zeros(n, dtype=np.float64)
    detalle["razones"] = np.array(["normal"] * n, dtype=object)
    return detalle


def _marcar(detalle, i, razon, score=5.0, **z):
    detalle["razones"][i] = razon
    detalle["score"][i] = float(score)
    for k, v in z.items():
        detalle[k][i] = float(v)


class TestCausaPrincipalCanal(unittest.TestCase):
    """La causa dominante debe salir de la razon y de la magnitud del z-score."""

    def test_amplitud_gana_cuando_su_z_es_mayor(self):
        z = {"z_std": 6.0, "z_ptp": 5.0, "z_hf": 1.0}
        self.assertEqual(P._causa_principal_canal("amplitud_extrema | ruido_hf", z), "amplitud")

    def test_ruido_hf_gana_cuando_z_hf_es_mayor(self):
        z = {"z_std": 1.0, "z_ptp": 1.2, "z_hf": 4.2}
        self.assertEqual(P._causa_principal_canal("amplitud_extrema | ruido_hf", z), "ruido_hf")

    def test_canal_bajo_usa_z_negativo(self):
        z = {"z_std": -5.0, "z_ptp": -6.0, "z_hf": 0.1}
        self.assertEqual(P._causa_principal_canal("canal_bajo_fuerte", z), "canal_bajo")

    def test_alfa_dominante(self):
        z = {"z_alpha": 7.5, "z_std": 1.0, "z_ptp": 1.0}
        self.assertEqual(P._causa_principal_canal("alfa_dominante_fuerte", z), "alfa_dominante")

    def test_espectro_anomalo_aplica_factor_0_9(self):
        # 5.0 * 0.9 = 4.5 frente a amplitud 4.6: debe ganar amplitud.
        z = {"z_dist": 5.0, "z_std": 4.6, "z_ptp": 0.0}
        self.assertEqual(P._causa_principal_canal("amplitud_alta | espectro_anomalo", z), "amplitud")
        # 6.0 * 0.9 = 5.4 > 4.6: ahora gana el espectro anomalo.
        z = {"z_dist": 6.0, "z_std": 4.6, "z_ptp": 0.0}
        self.assertEqual(P._causa_principal_canal("amplitud_alta | espectro_anomalo", z), "espectro_anomalo")

    def test_sin_razon_devuelve_espectro_anomalo(self):
        self.assertEqual(P._causa_principal_canal("", {}), "espectro_anomalo")
        self.assertEqual(P._causa_principal_canal(None, {}), "espectro_anomalo")

    def test_toda_causa_tiene_accion_definida(self):
        casos = {
            "amplitud_extrema": {"z_std": 5.0},
            "canal_bajo_fuerte": {"z_ptp": -5.0},
            "ruido_hf_fuerte": {"z_hf": 5.0},
            "alfa_dominante_fuerte": {"z_alpha": 5.0},
            "espectro_anomalo_fuerte": {"z_dist": 10.0},
            "": {},
        }
        for razon, z in casos.items():
            causa = P._causa_principal_canal(razon, z)
            self.assertIn(causa, P.ACCIONES_ADQUISICION, f"causa sin accion: {causa}")
            titulo, accion = P.ACCIONES_ADQUISICION[causa]
            self.assertTrue(titulo and accion)


class TestDetalleAtipicosDesdeReporte(unittest.TestCase):
    """Reconstruccion desde reporte_canales_atipicos.txt (caches antiguas)."""

    REPORTE = """Archivo: DCL 001.dat
Detector robusto de canales atipicos
==================================================

Criterio:
  Detector por reglas combinadas:
  - amplitud extrema
  - canal bajo/plano

No se detectaron canales atipicos fuertes.

Canales sospechosos (base 1):
7

Detalle por canal fuerte:
Canal 03 | score=9.10 | z_std=1.00 | z_rms=1.10 | z_ptp=1.20 | z_hf=4.20 | z_alpha=0.10 | z_delta=0.00 | z_dist=0.30 | razon=amplitud_extrema | ruido_hf
Canal 05 | score=7.50 | z_std=-5.00 | z_rms=-4.00 | z_ptp=-6.00 | z_hf=0.10 | z_alpha=0.00 | z_delta=0.00 | z_dist=0.10 | razon=canal_bajo_fuerte

Detalle por canal sospechoso:
Canal 07 | score=3.50 | z_std=0.50 | z_rms=0.40 | z_ptp=0.50 | z_hf=0.10
"""

    def _escribir(self, contenido):
        carpeta = tempfile.mkdtemp(prefix="neurox_reporte_")
        ruta = os.path.join(carpeta, "reporte_canales_atipicos.txt")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        return ruta

    def test_reporte_inexistente_devuelve_none(self):
        ruta = os.path.join(tempfile.mkdtemp(prefix="neurox_vacio_"), "no_existe.txt")
        self.assertIsNone(P.detalle_atipicos_desde_reporte(ruta, N))
        self.assertIsNone(P.detalle_atipicos_desde_reporte(None, N))

    def test_parsea_fuertes_y_sospechosos(self):
        detalle, fuertes, sospechosos = P.detalle_atipicos_desde_reporte(self._escribir(self.REPORTE), N)
        self.assertEqual(fuertes, [2, 4])
        self.assertEqual(sospechosos, [6])
        self.assertEqual(detalle["razones"][2], "amplitud_extrema | ruido_hf")
        self.assertEqual(detalle["razones"][4], "canal_bajo_fuerte")
        self.assertAlmostEqual(float(detalle["z_std"][4]), -5.0)
        self.assertAlmostEqual(float(detalle["score"][2]), 9.10)
        self.assertAlmostEqual(float(detalle["z_hf"][2]), 4.20)

    def test_sospechoso_infiere_razon_del_z_dominante(self):
        detalle, _f, _s = P.detalle_atipicos_desde_reporte(self._escribir(self.REPORTE), N)
        # Sin 'razon=' en el reporte: z_std es el dominante y es positivo.
        self.assertEqual(detalle["razones"][6], "amplitud_alta")

    def test_ignora_canales_fuera_de_rango(self):
        contenido = self.REPORTE.replace("Canal 05 |", "Canal 99 |")
        detalle, fuertes, _s = P.detalle_atipicos_desde_reporte(self._escribir(contenido), N)
        self.assertEqual(fuertes, [2])
        self.assertTrue(np.all(detalle["score"][:N] >= 0))

    def test_reporte_sin_canales_afectados_es_valido_y_vacio(self):
        contenido = (
            "Archivo: DCL 002.dat\n"
            "No se detectaron canales atipicos fuertes.\n\n"
            "No se detectaron canales sospechosos.\n\n"
            "Detalle por canal fuerte:\n  Ninguno\n\n"
            "Detalle por canal sospechoso:\n  Ninguno\n"
        )
        resultado = P.detalle_atipicos_desde_reporte(self._escribir(contenido), N)
        self.assertIsNotNone(resultado)
        detalle, fuertes, sospechosos = resultado
        self.assertEqual(fuertes, [])
        self.assertEqual(sospechosos, [])
        self.assertTrue(all(r == "normal" for r in detalle["razones"]))

    def test_reporte_vacio_no_rompe(self):
        detalle, fuertes, sospechosos = P.detalle_atipicos_desde_reporte(self._escribir(""), N)
        self.assertEqual((fuertes, sospechosos), ([], []))
        self.assertEqual(len(detalle["razones"]), N)


class TestDiagnosticarAdquisicion(unittest.TestCase):
    """Contrato de diagnosticar_adquisicion sobre detalles sinteticos."""

    def test_sin_atipicos_da_confiabilidad_alta(self):
        diag = P.diagnosticar_adquisicion(_detalle_vacio(), NOMBRES, [], [], snr_db=20.0)
        self.assertEqual(diag["confiabilidad_actual"], "ALTA")
        self.assertEqual(diag["electrodos_a_revisar"], [])
        self.assertIsNone(diag["patron_sistemico"])
        self.assertIsNone(diag["interferencia_red"])
        self.assertEqual(diag["acciones_protocolo"], [])
        self.assertEqual(diag["para_media"], {"electrodos": 0, "canales": []})
        self.assertEqual(diag["para_alta"], {"electrodos": 0, "canales": []})

    def test_patron_sistemico_al_20_por_ciento(self):
        detalle = _detalle_vacio()
        fuertes = list(range(20))
        for i in fuertes:
            _marcar(detalle, i, "ruido_hf_fuerte", score=100 - i, z_hf=5.0, z_std=0.5)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, fuertes, [], snr_db=20.0)
        self.assertIsNotNone(diag["patron_sistemico"])
        self.assertEqual(diag["patron_sistemico"]["causa"], "ruido_hf")
        self.assertEqual(diag["patron_sistemico"]["canales"], 20)
        self.assertIn("20 de 64 canales", diag["patron_sistemico"]["texto"])
        # El aviso de patron debe encabezar las acciones de protocolo.
        self.assertEqual(diag["acciones_protocolo"][0], diag["patron_sistemico"]["texto"])

    def test_sin_patron_sistemico_con_pocos_canales(self):
        detalle = _detalle_vacio()
        fuertes = list(range(5))
        for i in fuertes:
            _marcar(detalle, i, "ruido_hf", score=10 - i, z_hf=4.0)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, fuertes, [], snr_db=20.0)
        self.assertIsNone(diag["patron_sistemico"])
        self.assertEqual(len(diag["electrodos_a_revisar"]), 5)

    def test_interferencia_red_informa_snr_sin_red(self):
        diag = P.diagnosticar_adquisicion(
            _detalle_vacio(), NOMBRES, [], [], snr_db=-2.0,
            pct_ruido_red=64.0, snr_sin_red_db=12.0,
        )
        red = diag["interferencia_red"]
        self.assertIsNotNone(red)
        self.assertEqual(red["pct_ruido"], 64.0)
        self.assertEqual(red["snr_db"], -2.0)
        self.assertEqual(red["snr_sin_red_db"], 12.0)
        self.assertEqual(red["confiabilidad_sin_red"], "ALTA")
        self.assertIn("de -2.0 a 12.0 dB", red["texto"])
        self.assertIn("seria ALTA", red["texto"].replace("sería", "seria"))
        self.assertEqual(
            diag["acciones_protocolo"][0],
            P.ACCIONES_ADQUISICION["interferencia_red"][1],
        )

    def test_interferencia_red_en_el_umbral_dispara(self):
        diag = P.diagnosticar_adquisicion(
            _detalle_vacio(), NOMBRES, [], [], snr_db=10.0,
            pct_ruido_red=P.UMBRAL_INTERFERENCIA_RED_PCT, snr_sin_red_db=14.0,
        )
        self.assertIsNotNone(diag["interferencia_red"])

    def test_sin_interferencia_si_pct_bajo(self):
        diag = P.diagnosticar_adquisicion(
            _detalle_vacio(), NOMBRES, [], [], snr_db=10.0,
            pct_ruido_red=10.0, snr_sin_red_db=12.0,
        )
        self.assertIsNone(diag["interferencia_red"])
        self.assertNotIn(
            P.ACCIONES_ADQUISICION["interferencia_red"][1],
            diag["acciones_protocolo"],
        )

    def test_sin_datos_de_red_no_inventa_interferencia(self):
        diag = P.diagnosticar_adquisicion(
            _detalle_vacio(), NOMBRES, [], [], snr_db=10.0,
            pct_ruido_red=None, snr_sin_red_db=None,
        )
        self.assertIsNone(diag["interferencia_red"])

    def test_para_media_y_alta_calculan_el_minimo_real(self):
        # 32/64 canales fuertes (50 %) y SNR 16 dB: score = 3 - 3 = 0 -> NULA.
        detalle = _detalle_vacio()
        fuertes = list(range(32))
        for i in fuertes:
            _marcar(detalle, i, "ruido_hf", score=100 - i, z_hf=5.0)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, fuertes, [], snr_db=16.0)
        self.assertEqual(diag["confiabilidad_actual"], "NULA")

        # Con 20 corregidos quedan 12/64 = 18.75 % (sin penalizacion) -> MEDIA.
        self.assertEqual(diag["para_media"]["electrodos"], 20)
        self.assertEqual(len(diag["para_media"]["canales"]), 20)
        # Con 27 corregidos quedan 5/64 = 7.8 % (+2) -> ALTA.
        self.assertEqual(diag["para_alta"]["electrodos"], 27)

        # Invariante: el minimo reportado es el primero que alcanza el nivel.
        for objetivo, clave in (("MEDIA", "para_media"), ("ALTA", "para_alta")):
            k = diag[clave]["electrodos"]
            orden = {"NULA": 0, "BAJA": 1, "MEDIA": 2, "ALTA": 3}
            alcanzado = P._calcular_confiabilidad_metrica(16.0, (32 - k) / N * 100.0)
            anterior = P._calcular_confiabilidad_metrica(16.0, (32 - (k - 1)) / N * 100.0)
            self.assertGreaterEqual(orden[alcanzado], orden[objetivo])
            self.assertLess(orden[anterior], orden[objetivo])

    def test_para_media_none_cuando_el_snr_lo_impide(self):
        detalle = _detalle_vacio()
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, [], [], snr_db=-5.0)
        # aun con 0 % de atipicos el score maximo es 0 (NULA): el SNR manda.
        self.assertEqual(diag["confiabilidad_actual"], "NULA")
        self.assertIsNone(diag["para_media"])
        self.assertIsNone(diag["para_alta"])

    def test_fz_y_f2_no_disparan_patron_sistemico_actual(self):
        # Comportamiento documentado: 2/64 = 3.1 % no llega al umbral del 20 %.
        # La alerta de hardware de cohorte (FZ/F2 repetidos) es la tarea T-3.
        i_fz = NOMBRES.index("FZ")
        i_f2 = NOMBRES.index("F2")
        detalle = _detalle_vacio()
        for i, score in ((i_fz, 9.0), (i_f2, 8.0)):
            _marcar(detalle, i, "ruido_hf_fuerte", score=score, z_hf=6.0)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, [i_fz, i_f2], [], snr_db=12.0)
        self.assertIsNone(diag["patron_sistemico"])
        nombres_revisar = [e["canal"] for e in diag["electrodos_a_revisar"]]
        self.assertEqual(nombres_revisar, ["FZ", "F2"])
        self.assertEqual({e["causa"] for e in diag["electrodos_a_revisar"]}, {"ruido_hf"})

    def test_fuertes_antes_que_sospechosos_y_por_score(self):
        detalle = _detalle_vacio()
        _marcar(detalle, 3, "amplitud_alta", score=4.0, z_std=4.0)
        _marcar(detalle, 1, "ruido_hf_fuerte", score=9.0, z_hf=9.0)
        _marcar(detalle, 2, "ruido_hf_fuerte", score=7.0, z_hf=7.0)
        _marcar(detalle, 4, "alfa_dominante", score=3.0, z_alpha=3.0)
        diag = P.diagnosticar_adquisicion(
            detalle, NOMBRES, [3, 1, 2], [4], snr_db=15.0,
        )
        orden = [(e["canal"], e["nivel"]) for e in diag["electrodos_a_revisar"]]
        self.assertEqual(
            orden,
            [(NOMBRES[1], "atipico"), (NOMBRES[2], "atipico"),
             (NOMBRES[3], "atipico"), (NOMBRES[4], "sospechoso")],
        )

    def test_ficha_incluye_accion_y_descripcion(self):
        detalle = _detalle_vacio()
        _marcar(detalle, 9, "ruido_hf_fuerte", score=6.0, z_hf=6.0)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, [9], [], snr_db=8.0)
        ficha = diag["electrodos_a_revisar"][0]
        self.assertEqual(ficha["canal"], "FZ")
        self.assertEqual(ficha["nivel"], "atipico")
        self.assertEqual(ficha["causa"], "ruido_hf")
        self.assertEqual(ficha["descripcion"], P.ACCIONES_ADQUISICION["ruido_hf"][0])
        self.assertEqual(ficha["accion"], P.ACCIONES_ADQUISICION["ruido_hf"][1])
        self.assertAlmostEqual(ficha["score"], 6.0)

    def test_acciones_no_se_duplican(self):
        detalle = _detalle_vacio()
        _marcar(detalle, 1, "ruido_hf_fuerte", score=6.0, z_hf=6.0)
        _marcar(detalle, 2, "ruido_hf_fuerte", score=5.0, z_hf=5.0)
        diag = P.diagnosticar_adquisicion(detalle, NOMBRES, [1, 2], [], snr_db=6.0)
        self.assertEqual(len(diag["acciones_protocolo"]), len(set(diag["acciones_protocolo"])))


class TestRecomendacionesDesdeCache(unittest.TestCase):
    """Seleccion entre resumen_calidad.json y reconstruccion desde reporte."""

    def _cache(self, con_json=None, con_reporte=False):
        carpeta = tempfile.mkdtemp(prefix="neurox_cache_")
        os.makedirs(os.path.join(carpeta, "calidad_senal"), exist_ok=True)
        if con_json is not None:
            ruta = os.path.join(carpeta, "calidad_senal", "resumen_calidad.json")
            if isinstance(con_json, str):
                with open(ruta, "w", encoding="utf-8") as f:
                    f.write(con_json)
            else:
                with open(ruta, "w", encoding="utf-8") as f:
                    json.dump(con_json, f)
        if con_reporte:
            carpeta_rep = os.path.join(carpeta, "canales_atipicos")
            os.makedirs(carpeta_rep, exist_ok=True)
            with open(os.path.join(carpeta_rep, "reporte_canales_atipicos.txt"),
                      "w", encoding="utf-8") as f:
                f.write(TestDetalleAtipicosDesdeReporte.REPORTE)
        return carpeta

    def test_usa_el_json_cuando_existe(self):
        guardado = {"confiabilidad_actual": "MEDIA", "electrodos_a_revisar": [{"canal": "FZ"}]}
        cache = self._cache(con_json={"recomendaciones_adquisicion": guardado})
        self.assertEqual(P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES), guardado)

    def test_reconstruye_desde_reporte_en_cache_antigua(self):
        cache = self._cache(
            con_json={"snr": {"snr_global_db": 16.0, "pct_ruido_red": None, "snr_sin_red_db": None},
                      "canales_atipicos": ["FP2", "AF4"], "canales_sospechosos": ["F5"]},
            con_reporte=True,
        )
        diag = P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES)
        self.assertIsNotNone(diag)
        # Indices 3, 5 y 7 (base 1) -> FP2, AF4 y F5.
        self.assertEqual([e["canal"] for e in diag["electrodos_a_revisar"]], ["FP2", "AF4", "F5"])
        self.assertEqual(diag["electrodos_a_revisar"][-1]["nivel"], "sospechoso")

    def test_cache_sin_datos_devuelve_none(self):
        cache = self._cache()
        self.assertIsNone(P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES))

    def test_json_corrupto_cae_al_reporte(self):
        cache = self._cache(con_json="{esto no es json", con_reporte=True)
        diag = P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES)
        self.assertIsNotNone(diag)
        self.assertEqual(len(diag["electrodos_a_revisar"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
