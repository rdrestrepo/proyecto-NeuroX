# -*- coding: utf-8 -*-
"""
T-3 — Alerta de cohorte FZ/F2 (posible fallo de hardware del conector).

Cubre:

- lectura/escritura atomica y tolerante del historial (cohorte_fz_f2.json);
- actualizacion idempotente por registro (reprocesar no infla los conteos);
- reconstruccion del historial desde los resumen_calidad.json ya existentes;
- evaluacion de la alerta (ambos canales, umbral y minimo de registros);
- inyeccion de la alerta en las recomendaciones de cache;
- extension de config.json (fraccion y minimo entero);
- integracion con diagnosticar_adquisicion() sin duplicar acciones.

No procesa senales. Ejecutar desde la raiz:
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


def _detalle_vacio(n=N):
    detalle = {k: np.zeros(n, dtype=np.float64) for k in ("z_std", "z_ptp", "z_hf", "z_alpha", "z_dist")}
    detalle["score"] = np.zeros(n, dtype=np.float64)
    detalle["razones"] = np.array(["normal"] * n, dtype=object)
    return detalle


def _historial(registros):
    return {"version": 1, "actualizado": None, "registros": dict(registros)}


def _fila(canales=("FZ", "F2"), nivel="fuerte"):
    return {canal: nivel for canal in canales}


class _BaseCohorte(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="neurox_cohorte_")
        self.addCleanup(self._tmp.cleanup)
        self.base = self._tmp.name

    def _ruta(self, nombre="cohorte_fz_f2.json"):
        return os.path.join(self.base, nombre)

    def _guardar_json(self, ruta, datos, encoding="utf-8"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding=encoding, newline="\n") as f:
            json.dump(datos, f, ensure_ascii=False)
        return ruta

    def _cache(self, nombre="DCL_001", canales_atipicos=None, canales_sospechosos=None,
               recomendaciones=None, raiz=None):
        raiz = raiz or os.path.join(self.base, "CACHE")
        cache = os.path.join(raiz, nombre)
        resumen = {
            "canales_atipicos": list(canales_atipicos or []),
            "canales_sospechosos": list(canales_sospechosos or []),
        }
        if recomendaciones is not None:
            resumen["recomendaciones_adquisicion"] = recomendaciones
        self._guardar_json(
            os.path.join(cache, "calidad_senal", "resumen_calidad.json"), resumen
        )
        return cache

    def _historial_con_fallos(self, total=20, fz=True, f2=True, nivel="fuerte"):
        registros = {}
        for i in range(total):
            fila = {}
            if fz:
                fila["FZ"] = nivel
            if f2:
                fila["F2"] = nivel
            registros[f"REC_{i:03d}"] = fila
        return _historial(registros)


class TestRutasYConfig(_BaseCohorte):
    def test_ruta_historial_respeta_carpeta_base(self):
        self.assertEqual(
            P.ruta_historial_cohorte(self.base),
            os.path.join(os.path.abspath(self.base), P.NOMBRE_ARCHIVO_COHORTE),
        )

    def test_carpeta_base_desde_cache_valida(self):
        cache = os.path.join(self.base, "CACHE", "DCL_001")
        self.assertEqual(P._carpeta_base_desde_cache(cache), os.path.abspath(self.base))

    def test_carpeta_base_desde_cache_fuera_de_cache_es_none(self):
        self.assertIsNone(P._carpeta_base_desde_cache(os.path.join(self.base, "otra")))

    def test_defaults_de_cohorte(self):
        self.assertEqual(P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_cohorte_fz_f2"], 0.80)
        self.assertEqual(P.CONFIG_ADQUISICION_DEFAULTS["min_registros_cohorte"], 20)
        self.assertEqual(P.UMBRAL_PATRON_COHORTE_FZ_F2, 0.80)
        self.assertEqual(P.MIN_REGISTROS_COHORTE, 20)
        self.assertIsInstance(P.MIN_REGISTROS_COHORTE, int)

    def test_config_del_proyecto_declara_las_claves(self):
        ruta = os.path.join(RAIZ, "config.json")
        self.assertTrue(os.path.isfile(ruta))
        with open(ruta, "r", encoding="utf-8-sig") as f:
            datos = json.load(f)
        seccion = datos["adquisicion"]
        self.assertEqual(seccion["umbral_patron_cohorte_fz_f2"], 0.8)
        self.assertEqual(seccion["min_registros_cohorte"], 20)
        self.assertEqual(seccion["umbral_interferencia_red_pct"], 30.0)
        self.assertEqual(seccion["umbral_patron_sistemico"], 0.2)

        config, origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(advertencias, [])
        self.assertEqual(config["umbral_patron_cohorte_fz_f2"], 0.8)
        self.assertEqual(config["min_registros_cohorte"], 20)


class TestCargaHistorial(_BaseCohorte):
    def test_archivo_ausente_devuelve_vacio(self):
        historial, origen, avisos = P.cargar_historial_cohorte(self._ruta())
        self.assertEqual(historial["registros"], {})
        self.assertIsNone(origen)
        self.assertEqual(avisos, [])

    def test_json_corrupto_advierte_y_devuelve_vacio(self):
        ruta = self._ruta()
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("{esto no es json")
        historial, origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(historial["registros"], {})
        self.assertIsNone(origen)
        self.assertEqual(len(avisos), 1)

    def test_raiz_que_no_es_objeto_advierte(self):
        ruta = self._guardar_json(self._ruta(), [1, 2, 3])
        historial, _origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(historial["registros"], {})
        self.assertTrue(avisos)

    def test_sin_registros_validos_advierte(self):
        ruta = self._guardar_json(self._ruta(), {"registros": ["FZ", "F2"]})
        historial, _origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(historial["registros"], {})
        self.assertTrue(avisos)

    def test_normaliza_solo_fz_f2_y_niveles_validos(self):
        ruta = self._guardar_json(
            self._ruta(),
            {
                "registros": {
                    "A": {"FZ": "fuerte", "F2": "sospechoso", "CZ": "fuerte", "AF4": "fuerte"},
                    "B": {"FZ": "atipico", "F2": "inventado"},
                    "C": {"FZ": None, "F2": 3},
                    "D": "no-es-dict",
                    "E": {},
                }
            },
        )
        historial, origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(avisos, [])
        self.assertEqual(historial["registros"]["A"], {"FZ": "fuerte", "F2": "sospechoso"})
        self.assertEqual(historial["registros"]["B"], {"FZ": "fuerte"})
        self.assertEqual(historial["registros"]["C"], {})
        self.assertNotIn("D", historial["registros"])
        self.assertEqual(historial["registros"]["E"], {})

    def test_acepta_bom(self):
        ruta = self._ruta()
        with open(ruta, "w", encoding="utf-8-sig", newline="\n") as f:
            json.dump({"registros": {"A": {"FZ": "fuerte"}}}, f)
        historial, _origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(avisos, [])
        self.assertEqual(historial["registros"]["A"], {"FZ": "fuerte"})


class TestActualizacion(_BaseCohorte):
    def test_registra_fuertes_y_sospechosos(self):
        historial = _historial({})
        P.actualizar_historial_cohorte(
            historial, "REC", ["FZ"], ["F2", "CZ"]
        )
        self.assertEqual(historial["registros"]["REC"], {"FZ": "fuerte", "F2": "sospechoso"})

    def test_fuerte_prevalece_sobre_sospechoso(self):
        historial = _historial({})
        P.actualizar_historial_cohorte(historial, "REC", ["FZ", "F2"], ["FZ", "F2"])
        self.assertEqual(historial["registros"]["REC"], {"FZ": "fuerte", "F2": "fuerte"})

    def test_idempotente_por_registro(self):
        historial = _historial({})
        for _ in range(3):
            P.actualizar_historial_cohorte(historial, "REC", ["FZ", "F2"], [])
        self.assertEqual(len(historial["registros"]), 1)
        self.assertEqual(historial["registros"]["REC"], {"FZ": "fuerte", "F2": "fuerte"})

    def test_reemplaza_entrada_previa(self):
        historial = _historial({"REC": {"FZ": "fuerte", "F2": "fuerte"}})
        P.actualizar_historial_cohorte(historial, "REC", [], [])
        self.assertEqual(historial["registros"]["REC"], {})

    def test_ignora_canales_fuera_de_interes(self):
        historial = _historial({})
        P.actualizar_historial_cohorte(historial, "REC", ["CZ", "AF4"], ["O1"])
        self.assertEqual(historial["registros"]["REC"], {})

    def test_tolera_listas_invalidas(self):
        historial = _historial({})
        P.actualizar_historial_cohorte(historial, "REC", 123, {"FZ"})
        self.assertEqual(historial["registros"]["REC"], {"FZ": "sospechoso"})


class TestGuardado(_BaseCohorte):
    def test_roundtrip(self):
        ruta = self._ruta()
        historial = self._historial_con_fallos(total=2)
        self.assertTrue(P.guardar_historial_cohorte(historial, ruta))
        self.assertTrue(os.path.isfile(ruta))
        cargado, origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(avisos, [])
        self.assertEqual(set(cargado["registros"]), {"REC_000", "REC_001"})

    def test_escribe_ordenado_y_con_marca_de_tiempo(self):
        ruta = self._ruta()
        historial = _historial({"Z": {"FZ": "fuerte"}, "A": {"F2": "fuerte"}})
        self.assertTrue(P.guardar_historial_cohorte(historial, ruta))
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
        self.assertEqual(list(datos["registros"]), ["A", "Z"])
        self.assertIsInstance(datos["actualizado"], str)
        self.assertTrue(datos["actualizado"])

    def test_no_deja_archivo_temporal(self):
        ruta = self._ruta()
        self.assertTrue(P.guardar_historial_cohorte(_historial({"A": {}}), ruta))
        sobrantes = [n for n in os.listdir(self.base) if n.endswith(".tmp")]
        self.assertEqual(sobrantes, [])

    def test_fallo_de_serializacion_conserva_el_original(self):
        ruta = self._ruta()
        self.assertTrue(P.guardar_historial_cohorte(_historial({"A": {"FZ": "fuerte"}}), ruta))
        with open(ruta, "r", encoding="utf-8") as f:
            antes = f.read()

        corrupto = _historial({"B": {"FZ": {"no", "serializable"}}})
        self.assertFalse(P.guardar_historial_cohorte(corrupto, ruta))
        with open(ruta, "r", encoding="utf-8") as f:
            despues = f.read()
        self.assertEqual(antes, despues)

    def test_rechaza_historial_invalido_sin_tocar_el_original(self):
        ruta = self._ruta()
        self.assertTrue(P.guardar_historial_cohorte(_historial({"A": {}}), ruta))
        with open(ruta, "r", encoding="utf-8") as f:
            antes = f.read()
        self.assertFalse(P.guardar_historial_cohorte(None, ruta))
        self.assertFalse(P.guardar_historial_cohorte({"registros": []}, ruta))
        with open(ruta, "r", encoding="utf-8") as f:
            self.assertEqual(antes, f.read())


class TestReconstruccion(_BaseCohorte):
    def test_reconstruye_desde_los_resumenes_de_cache(self):
        self._cache("DCL_001", ["FZ", "F2"], ["CZ"])
        self._cache("DCL_002", ["FZ"], ["F2"])
        self._cache("DCL_003", ["CZ"], [])
        historial = P.reconstruir_historial_desde_cache(self.base)
        self.assertEqual(set(historial["registros"]), {"DCL_001", "DCL_002", "DCL_003"})
        self.assertEqual(historial["registros"]["DCL_001"], {"FZ": "fuerte", "F2": "fuerte"})
        self.assertEqual(historial["registros"]["DCL_002"], {"FZ": "fuerte", "F2": "sospechoso"})
        self.assertEqual(historial["registros"]["DCL_003"], {})

    def test_ignora_cache_sin_json_o_con_json_corrupto(self):
        os.makedirs(os.path.join(self.base, "CACHE", "VACIO"), exist_ok=True)
        corrupto = os.path.join(self.base, "CACHE", "CORRUPTO", "calidad_senal")
        os.makedirs(corrupto, exist_ok=True)
        with open(os.path.join(corrupto, "resumen_calidad.json"), "w", encoding="utf-8") as f:
            f.write("{no")
        self._cache("BUENO", ["FZ"], [])
        historial = P.reconstruir_historial_desde_cache(self.base)
        self.assertEqual(set(historial["registros"]), {"BUENO"})

    def test_sin_carpeta_cache_devuelve_vacio(self):
        historial = P.reconstruir_historial_desde_cache(self.base)
        self.assertEqual(historial["registros"], {})

    def test_asegurar_no_crea_archivo_si_no_hay_caches(self):
        ruta = self._ruta()
        historial, origen, avisos = P.asegurar_historial_cohorte(ruta=ruta, carpeta_base=self.base)
        self.assertEqual(historial["registros"], {})
        self.assertIsNone(origen)
        self.assertEqual(avisos, [])
        self.assertFalse(os.path.exists(ruta))

    def test_estructura_invalida_no_se_considera_origen(self):
        ruta = self._guardar_json(self._ruta(), {"registros": []})
        _historial, origen, avisos = P.cargar_historial_cohorte(ruta)
        self.assertIsNone(origen)
        self.assertTrue(avisos)

    def test_asegurar_reconstruye_si_el_archivo_esta_corrupto(self):
        self._cache("DCL_001", ["FZ", "F2"], [])
        ruta = self._ruta()
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("{corrupto")
        historial, origen, avisos = P.asegurar_historial_cohorte(
            ruta=ruta, carpeta_base=self.base
        )
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(set(historial["registros"]), {"DCL_001"})
        self.assertTrue(any("reconstruido" in aviso for aviso in avisos))

        cargado, origen2, avisos2 = P.cargar_historial_cohorte(ruta)
        self.assertEqual(origen2, os.path.abspath(ruta))
        self.assertEqual(avisos2, [])
        self.assertEqual(set(cargado["registros"]), {"DCL_001"})

    def test_asegurar_reconstruye_y_reutiliza(self):
        self._cache("DCL_001", ["FZ", "F2"], [])
        ruta = self._ruta()
        historial, origen, _avisos = P.asegurar_historial_cohorte(ruta=ruta, carpeta_base=self.base)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertTrue(os.path.isfile(ruta))
        self.assertEqual(set(historial["registros"]), {"DCL_001"})

        # Un cache nuevo no se incorpora hasta que se registre explicitamente.
        self._cache("DCL_002", ["FZ", "F2"], [])
        historial2, origen2, _ = P.asegurar_historial_cohorte(ruta=ruta, carpeta_base=self.base)
        self.assertEqual(origen2, os.path.abspath(ruta))
        self.assertEqual(set(historial2["registros"]), {"DCL_001"})


class TestEvaluacion(_BaseCohorte):
    def test_minimo_de_registros_no_alcanzado(self):
        historial = self._historial_con_fallos(total=19)
        self.assertIsNone(P.evaluar_alerta_cohorte(historial))

    def test_un_solo_canal_alto_no_alerta(self):
        historial = self._historial_con_fallos(total=20, f2=False)
        self.assertIsNone(P.evaluar_alerta_cohorte(historial))

    def test_ambos_canales_altos_alerta(self):
        historial = self._historial_con_fallos(total=20)
        alerta = P.evaluar_alerta_cohorte(historial)
        self.assertIsNotNone(alerta)
        self.assertEqual(alerta["alerta"], "HARDWARE_CONECTOR_FZ_F2")
        self.assertEqual(alerta["registros"], 20)
        self.assertEqual(alerta["canales"]["FZ"]["pct"], 1.0)
        self.assertEqual(alerta["canales"]["F2"]["pct"], 1.0)
        self.assertIn("20 registros", alerta["texto"])
        self.assertIn("100 %", alerta["texto"])
        self.assertIn("Cohorte FZ/F2", alerta["texto_corto"])
        self.assertIn("20 registros", alerta["texto_corto"])

    def test_limite_exacto_del_umbral(self):
        registros = {}
        for i in range(20):
            fila = {}
            if i < 16:
                fila = {"FZ": "fuerte", "F2": "fuerte"}
            registros[f"R{i}"] = fila
        self.assertIsNotNone(P.evaluar_alerta_cohorte(_historial(registros)))

    def test_por_debajo_del_umbral(self):
        registros = {}
        for i in range(20):
            fila = {}
            if i < 15:
                fila = {"FZ": "fuerte", "F2": "fuerte"}
            registros[f"R{i}"] = fila
        self.assertIsNone(P.evaluar_alerta_cohorte(_historial(registros)))

    def test_sospechosos_no_cuentan(self):
        historial = self._historial_con_fallos(total=20, nivel="sospechoso")
        self.assertIsNone(P.evaluar_alerta_cohorte(historial))

    def test_umbral_y_minimo_personalizados(self):
        registros = {}
        for i in range(5):
            registros[f"R{i}"] = {"FZ": "fuerte", "F2": "fuerte"}
        alerta = P.evaluar_alerta_cohorte(_historial(registros), umbral=0.6, min_registros=3)
        self.assertIsNotNone(alerta)
        self.assertEqual(alerta["umbral"], 0.6)
        self.assertEqual(alerta["min_registros"], 3)

    def test_historial_invalido_devuelve_none(self):
        self.assertIsNone(P.evaluar_alerta_cohorte(None))
        self.assertIsNone(P.evaluar_alerta_cohorte({"registros": []}))
        self.assertIsNone(P.evaluar_alerta_cohorte({"registros": {}}, min_registros=0))

    def test_texto_incluye_porcentajes_por_canal(self):
        registros = {}
        for i in range(10):
            fila = {"FZ": "fuerte"} if i < 9 else {}
            if i < 8:
                fila["F2"] = "fuerte"
            registros[f"R{i}"] = fila
        alerta = P.evaluar_alerta_cohorte(_historial(registros), umbral=0.8, min_registros=5)
        self.assertIsNotNone(alerta)
        self.assertIn("90 %", alerta["texto"])
        self.assertIn("80 %", alerta["texto"])
        self.assertEqual(alerta["canales"]["FZ"]["registros_fuertes"], 9)
        self.assertEqual(alerta["canales"]["F2"]["registros_fuertes"], 8)


class TestRegistro(_BaseCohorte):
    def test_registrar_crea_el_archivo_y_devuelve_alerta(self):
        ruta = self._ruta()
        alerta = P.registrar_y_evaluar_cohorte(
            ruta, "REC_001", ["FZ", "F2"], [], min_registros=1, umbral=0.8
        )
        self.assertIsNotNone(alerta)
        self.assertEqual(alerta["registros"], 1)
        self.assertTrue(os.path.isfile(ruta))

    def test_registrar_es_idempotente(self):
        ruta = self._ruta()
        historial = self._historial_con_fallos(total=19)
        P.guardar_historial_cohorte(historial, ruta)

        for _ in range(2):
            P.registrar_y_evaluar_cohorte(
                ruta, "REC_019", ["FZ", "F2"], [], min_registros=20, umbral=0.8
            )

        cargado, _origen, _avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(len(cargado["registros"]), 20)
        self.assertEqual(sum(1 for f in cargado["registros"].values() if f.get("FZ") == "fuerte"), 20)

    def test_registrar_reemplaza_la_entrada_previa(self):
        ruta = self._ruta()
        P.guardar_historial_cohorte(_historial({"REC": {"FZ": "fuerte", "F2": "fuerte"}}), ruta)
        P.registrar_y_evaluar_cohorte(ruta, "REC", [], [], min_registros=1)
        cargado, _origen, _avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(cargado["registros"]["REC"], {})

    def test_registrar_actualiza_el_historial_existente(self):
        ruta = self._ruta()
        P.guardar_historial_cohorte(_historial({"VIEJO": {"FZ": "fuerte"}}), ruta)
        P.registrar_y_evaluar_cohorte(ruta, "NUEVO", ["FZ", "F2"], [], min_registros=1)
        cargado, _origen, _avisos = P.cargar_historial_cohorte(ruta)
        self.assertEqual(set(cargado["registros"]), {"VIEJO", "NUEVO"})


class TestIntegracion(_BaseCohorte):
    def test_diagnosticar_incluye_la_alerta_sin_duplicar_acciones(self):
        alerta = {
            "alerta": "HARDWARE_CONECTOR_FZ_F2",
            "registros": 20,
            "texto": "FZ y F2 fallan en la cohorte.",
        }
        diag = P.diagnosticar_adquisicion(
            _detalle_vacio(), NOMBRES, [], [], snr_db=None,
            alerta_cohorte=alerta,
        )
        self.assertEqual(diag["alerta_cohorte"], alerta)
        self.assertNotIn(alerta["texto"], diag["acciones_protocolo"])

    def test_recomendaciones_inyecta_la_alerta_con_ruta_explicita(self):
        cache = self._cache("DCL_001", ["FZ", "F2"], [],
                            recomendaciones={"confiabilidad_actual": "MEDIA",
                                             "electrodos_a_revisar": []})
        ruta_hist = self._ruta()
        P.guardar_historial_cohorte(self._historial_con_fallos(total=20), ruta_hist)

        rec = P.recomendaciones_adquisicion_desde_cache(
            cache, NOMBRES, ruta_historial_cohorte=ruta_hist
        )
        self.assertIsNotNone(rec)
        self.assertIn("alerta_cohorte", rec)
        self.assertEqual(rec["alerta_cohorte"]["alerta"], "HARDWARE_CONECTOR_FZ_F2")

        # El JSON de la cache no se modifica: la alerta es contextual.
        with open(os.path.join(cache, "calidad_senal", "resumen_calidad.json"),
                  "r", encoding="utf-8") as f:
            guardado = json.load(f)
        self.assertNotIn("alerta_cohorte", guardado["recomendaciones_adquisicion"])

    def test_recomendaciones_deriva_la_base_desde_la_cache(self):
        cache = self._cache("DCL_001", ["FZ", "F2"], [],
                            recomendaciones={"electrodos_a_revisar": []})
        P.guardar_historial_cohorte(
            self._historial_con_fallos(total=20), self._ruta()
        )
        rec = P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES)
        self.assertIn("alerta_cohorte", rec)

    def test_recomendaciones_fuera_de_cache_no_inyecta_ni_crea_archivos(self):
        cache = os.path.join(self.base, "suelta")
        self._guardar_json(
            os.path.join(cache, "calidad_senal", "resumen_calidad.json"),
            {"recomendaciones_adquisicion": {"electrodos_a_revisar": []}},
        )
        rec = P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES)
        self.assertNotIn("alerta_cohorte", rec)
        self.assertFalse(os.path.exists(self._ruta()))

    def test_recomendaciones_reconstruidas_tambien_reciben_la_alerta(self):
        cache = self._cache("DCL_001", ["FP2"], [], recomendaciones=None)
        reporte = os.path.join(cache, "canales_atipicos")
        os.makedirs(reporte, exist_ok=True)
        with open(os.path.join(reporte, "reporte_canales_atipicos.txt"),
                  "w", encoding="utf-8") as f:
            f.write("Detalle por canal fuerte:\n")
            f.write("Canal 03 | score=5.0 | z_std=4.0 | z_ptp=4.5 | z_hf=0.5 | razon=amplitud_extrema\n")
        P.guardar_historial_cohorte(self._historial_con_fallos(total=20), self._ruta())

        rec = P.recomendaciones_adquisicion_desde_cache(cache, NOMBRES)
        self.assertIsNotNone(rec)
        self.assertIn("alerta_cohorte", rec)
        self.assertEqual([e["canal"] for e in rec["electrodos_a_revisar"]], ["FP2"])


class TestConfigCohorte(_BaseCohorte):
    def _config(self, seccion):
        return self._guardar_json(
            os.path.join(self.base, "config.json"), {"adquisicion": seccion}
        )

    def test_min_registros_acepta_entero(self):
        ruta = self._config({"min_registros_cohorte": 25})
        config, _origen, avisos = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["min_registros_cohorte"], 25)
        self.assertIsInstance(config["min_registros_cohorte"], int)
        self.assertEqual(avisos, [])

    def test_min_registros_decimal_se_rechaza(self):
        ruta = self._config({"min_registros_cohorte": 20.5})
        config, _origen, avisos = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["min_registros_cohorte"], 20)
        self.assertTrue(avisos)

    def test_min_registros_fuera_de_rango_se_rechaza(self):
        ruta = self._config({"min_registros_cohorte": 0})
        config, _origen, avisos = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["min_registros_cohorte"], 20)
        self.assertTrue(avisos)

    def test_umbral_de_cohorte_fuera_de_rango_se_rechaza(self):
        ruta = self._config({"umbral_patron_cohorte_fz_f2": 1.5})
        config, _origen, avisos = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_patron_cohorte_fz_f2"], 0.80)
        self.assertTrue(avisos)

    def test_umbral_de_cohorte_valido(self):
        ruta = self._config({"umbral_patron_cohorte_fz_f2": 0.65})
        config, _origen, avisos = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_patron_cohorte_fz_f2"], 0.65)
        self.assertEqual(avisos, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
