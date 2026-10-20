# -*- coding: utf-8 -*-
"""
T-2 — Umbrales de adquisicion configurables (config.json).

Verifica el cargador que permite ajustar los umbrales del diagnostico de
adquisicion sin tocar codigo:

- defaults identicos a los valores historicos (30 % y 0.20);
- JSON valido, parcial, vacio, con BOM y con claves de documentacion "_";
- errores tolerados: JSON corrupto, raiz o seccion que no son objetos,
  claves desconocidas, tipos incorrectos, NaN/Infinity y valores fuera de
  rango (siempre se cae al default y se anota una advertencia);
- las constantes que usa `diagnosticar_adquisicion()` quedan cableadas al
  config efectivo.

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


class _BaseConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="neurox_config_")
        self.addCleanup(self._tmp.cleanup)

    def _ruta(self, nombre="config.json"):
        return os.path.join(self._tmp.name, nombre)

    def _escribir(self, contenido, encoding="utf-8", nombre="config.json"):
        ruta = self._ruta(nombre)
        with open(ruta, "w", encoding=encoding, newline="\n") as f:
            f.write(contenido)
        return ruta

    def _json(self, datos, encoding="utf-8", nombre="config.json"):
        return self._escribir(json.dumps(datos, ensure_ascii=False), encoding, nombre)


class TestIntegridadDelEsquema(_BaseConfig):
    def test_rangos_cubren_las_mismas_claves_que_los_defaults(self):
        self.assertEqual(
            set(P.CONFIG_ADQUISICION_RANGOS), set(P.CONFIG_ADQUISICION_DEFAULTS)
        )

    def test_defaults_dentro_de_su_rango(self):
        for clave, defecto in P.CONFIG_ADQUISICION_DEFAULTS.items():
            minimo, maximo = P.CONFIG_ADQUISICION_RANGOS[clave]
            self.assertLessEqual(minimo, defecto, clave)
            self.assertLessEqual(defecto, maximo, clave)

    def test_defaults_conservan_los_valores_historicos(self):
        # Regresion: T-2 no cambia ningun umbral, solo lo hace configurable.
        self.assertEqual(P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"], 30.0)
        self.assertEqual(P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_sistemico"], 0.20)

    def test_constantes_efectivas_salen_del_config(self):
        self.assertEqual(
            P.UMBRAL_INTERFERENCIA_RED_PCT,
            P.CONFIG_ADQUISICION["umbral_interferencia_red_pct"],
        )
        self.assertEqual(
            P.UMBRAL_PATRON_SISTEMICO,
            P.CONFIG_ADQUISICION["umbral_patron_sistemico"],
        )

    def test_config_del_proyecto_existe_y_es_valido(self):
        ruta = P.localizar_config_adquisicion()
        self.assertIsNotNone(ruta, "falta config.json en la raiz del proyecto")
        self.assertTrue(os.path.isfile(ruta))
        config, origen, advertencias = P.cargar_config_adquisicion()
        self.assertEqual(advertencias, [])
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(set(config), set(P.CONFIG_ADQUISICION_DEFAULTS))
        for clave, valor in config.items():
            minimo, maximo = P.CONFIG_ADQUISICION_RANGOS[clave]
            self.assertTrue(minimo <= valor <= maximo, clave)


class TestCargaBasica(_BaseConfig):
    def test_sin_archivo_usa_defaults_sin_advierte(self):
        config, origen, advertencias = P.cargar_config_adquisicion(self._ruta("no_existe.json"))
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertIsNone(origen)
        self.assertEqual(advertencias, [])

    def test_config_valido_sobrescribe(self):
        ruta = self._json({"adquisicion": {
            "umbral_interferencia_red_pct": 12.5,
            "umbral_patron_sistemico": 0.35,
        }})
        config, origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_interferencia_red_pct"], 12.5)
        self.assertEqual(config["umbral_patron_sistemico"], 0.35)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(advertencias, [])

    def test_config_parcial_completa_con_defaults(self):
        ruta = self._json({"adquisicion": {"umbral_patron_sistemico": 0.5}})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_patron_sistemico"], 0.5)
        self.assertEqual(
            config["umbral_interferencia_red_pct"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
        )
        self.assertEqual(advertencias, [])

    def test_objeto_vacio_usa_defaults_sin_advierte(self):
        ruta = self._json({})
        config, origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(advertencias, [])

    def test_claves_de_documentacion_se_ignoran(self):
        ruta = self._json({"adquisicion": {
            "_ayuda": "texto libre",
            "_umbral_patron_sistemico": "0.20 = 20 %",
            "umbral_patron_sistemico": 0.3,
        }})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_patron_sistemico"], 0.3)
        self.assertEqual(advertencias, [])

    def test_bom_utf8_aceptado(self):
        ruta = self._json({"adquisicion": {"umbral_interferencia_red_pct": 25.0}},
                          encoding="utf-8-sig")
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_interferencia_red_pct"], 25.0)
        self.assertEqual(advertencias, [])


class TestBusquedaDeArchivo(_BaseConfig):
    """Rutas de config.json en desarrollo y dentro del .exe empaquetado."""

    def setUp(self):
        super().setUp()
        self._estado = {
            nombre: (hasattr(sys, nombre), getattr(sys, nombre, None))
            for nombre in ("frozen", "_MEIPASS", "executable")
        }
        self.addCleanup(self._restaurar_sys)

    def _restaurar_sys(self):
        for nombre, (existia, valor) in self._estado.items():
            if existia:
                setattr(sys, nombre, valor)
            else:
                try:
                    delattr(sys, nombre)
                except AttributeError:
                    pass

    def _simular_frozen(self):
        exe_dir = os.path.join(self._tmp.name, "dist", "NeuroX")
        internal = os.path.join(exe_dir, "_internal")
        os.makedirs(internal)
        sys.frozen = True
        sys.executable = os.path.join(exe_dir, "NeuroX.exe")
        sys._MEIPASS = internal
        return exe_dir, internal

    def test_modo_desarrollo_usa_carpeta_del_pipeline(self):
        self.assertFalse(getattr(sys, "frozen", False))
        carpetas = P._carpetas_busqueda_config()
        self.assertEqual(carpetas, [os.path.dirname(os.path.abspath(P.__file__))])

    def test_en_exe_prioriza_carpeta_del_ejecutable(self):
        exe_dir, internal = self._simular_frozen()
        self.assertEqual(P._carpetas_busqueda_config(), [exe_dir, internal])

    def test_localiza_config_junto_al_exe_antes_que_el_empaquetado(self):
        exe_dir, internal = self._simular_frozen()
        ruta_exe = os.path.join(exe_dir, "config.json")
        ruta_internal = os.path.join(internal, "config.json")
        with open(ruta_exe, "w", encoding="utf-8") as f:
            f.write('{"adquisicion": {"umbral_patron_sistemico": 0.4}}')
        with open(ruta_internal, "w", encoding="utf-8") as f:
            f.write('{"adquisicion": {"umbral_patron_sistemico": 0.6}}')
        self.assertEqual(P.localizar_config_adquisicion(), ruta_exe)
        config, origen, _adv = P.cargar_config_adquisicion()
        self.assertEqual(config["umbral_patron_sistemico"], 0.4)
        self.assertEqual(origen, os.path.abspath(ruta_exe))

    def test_cae_al_config_empaquetado_si_falta_junto_al_exe(self):
        _exe_dir, internal = self._simular_frozen()
        ruta_internal = os.path.join(internal, "config.json")
        with open(ruta_internal, "w", encoding="utf-8") as f:
            f.write('{"adquisicion": {"umbral_patron_sistemico": 0.6}}')
        self.assertEqual(P.localizar_config_adquisicion(), ruta_internal)
        config, _origen, _adv = P.cargar_config_adquisicion()
        self.assertEqual(config["umbral_patron_sistemico"], 0.6)


class TestErroresTolerados(_BaseConfig):
    def test_json_corrupto_usa_defaults_y_advierte(self):
        ruta = self._escribir("{esto no es json")
        config, origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertIsNone(origen)
        self.assertEqual(len(advertencias), 1)
        self.assertIn("No se pudo leer", advertencias[0])

    def test_raiz_no_objeto_usa_defaults_y_advierte(self):
        ruta = self._json([])
        config, origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(origen, os.path.abspath(ruta))
        self.assertEqual(len(advertencias), 1)
        self.assertIn("objeto JSON en la raiz", advertencias[0])

    def test_seccion_no_objeto_usa_defaults_y_advierte(self):
        ruta = self._json({"adquisicion": 5})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(len(advertencias), 1)
        self.assertIn("debe ser un objeto JSON", advertencias[0])

    def test_falta_seccion_con_claves_advierte(self):
        ruta = self._json({"umbral_patron_sistemico": 0.3})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(len(advertencias), 1)
        self.assertIn("no tiene la seccion 'adquisicion'", advertencias[0])

    def test_falta_seccion_solo_documentacion_no_advierte(self):
        ruta = self._json({"_ayuda": "solo documentacion"})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(advertencias, [])

    def test_clave_desconocida_se_ignora_y_advierte(self):
        ruta = self._json({"adquisicion": {
            "umbral_interferencia_red": 10.0,
            "umbral_patron_sistemico": 0.25,
        }})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_patron_sistemico"], 0.25)
        self.assertEqual(
            config["umbral_interferencia_red_pct"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("Clave desconocida", advertencias[0])

    def test_texto_rechazado(self):
        ruta = self._json({"adquisicion": {"umbral_interferencia_red_pct": "30"}})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_interferencia_red_pct"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("se esperaba un numero", advertencias[0])

    def test_booleano_rechazado(self):
        # bool es subclase de int: debe rechazarse explicitamente.
        ruta = self._json({"adquisicion": {"umbral_patron_sistemico": True}})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_patron_sistemico"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_sistemico"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("bool", advertencias[0])

    def test_nulo_rechazado(self):
        ruta = self._json({"adquisicion": {"umbral_patron_sistemico": None}})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_patron_sistemico"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_sistemico"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("NoneType", advertencias[0])

    def test_nan_rechazado(self):
        ruta = self._escribir(
            '{"adquisicion": {"umbral_interferencia_red_pct": NaN}}'
        )
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_interferencia_red_pct"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("no es finito", advertencias[0])

    def test_infinito_rechazado(self):
        ruta = self._escribir(
            '{"adquisicion": {"umbral_patron_sistemico": Infinity}}'
        )
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_patron_sistemico"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_sistemico"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("no es finito", advertencias[0])

    def test_interferencia_fuera_de_rango_rechazada(self):
        for invalido in (-1.0, 150.0):
            ruta = self._json({"adquisicion": {
                "umbral_interferencia_red_pct": invalido,
            }})
            config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
            self.assertEqual(
                config["umbral_interferencia_red_pct"],
                P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
                invalido,
            )
            self.assertEqual(len(advertencias), 1, invalido)
            self.assertIn("fuera del rango", advertencias[0])

    def test_patron_fuera_de_rango_rechazado(self):
        for invalido in (-0.1, 1.5, 20.0):
            ruta = self._json({"adquisicion": {
                "umbral_patron_sistemico": invalido,
            }})
            config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
            self.assertEqual(
                config["umbral_patron_sistemico"],
                P.CONFIG_ADQUISICION_DEFAULTS["umbral_patron_sistemico"],
                invalido,
            )
            self.assertEqual(len(advertencias), 1, invalido)
            self.assertIn("fuera del rango", advertencias[0])

    def test_entero_desmesurado_rechazado(self):
        # Un int enorme (400 digitos) desborda float(): debe caer al default.
        ruta = self._escribir(
            '{"adquisicion": {"umbral_interferencia_red_pct": 1' + "0" * 400 + "}}"
        )
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(
            config["umbral_interferencia_red_pct"],
            P.CONFIG_ADQUISICION_DEFAULTS["umbral_interferencia_red_pct"],
        )
        self.assertEqual(len(advertencias), 1)
        self.assertIn("no se puede representar", advertencias[0])

    def test_ruta_que_es_directorio_advierte(self):
        config, origen, advertencias = P.cargar_config_adquisicion(self._tmp.name)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertIsNone(origen)
        self.assertEqual(len(advertencias), 1)
        self.assertIn("No se pudo leer", advertencias[0])

    def test_dos_errores_generan_dos_advertencias(self):
        ruta = self._json({"adquisicion": {
            "umbral_interferencia_red_pct": "malo",
            "umbral_patron_sistemico": 1.5,
        }})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config, P.CONFIG_ADQUISICION_DEFAULTS)
        self.assertEqual(len(advertencias), 2)

    def test_limites_inclusivos_aceptados(self):
        ruta = self._json({"adquisicion": {
            "umbral_interferencia_red_pct": 0.0,
            "umbral_patron_sistemico": 1.0,
        }})
        config, _origen, advertencias = P.cargar_config_adquisicion(ruta)
        self.assertEqual(config["umbral_interferencia_red_pct"], 0.0)
        self.assertEqual(config["umbral_patron_sistemico"], 1.0)
        self.assertEqual(advertencias, [])


class TestIntegracionConDiagnostico(_BaseConfig):
    def test_patron_sistemico_respeta_la_constante_del_modulo(self):
        # 20 de 64 canales = 31.25 %: con 0.20 avisa; con 0.50 no.
        detalle = {k: np.zeros(len(NOMBRES)) for k in
                   ("z_std", "z_ptp", "z_hf", "z_alpha", "z_dist", "score")}
        detalle["razones"] = np.array(["normal"] * len(NOMBRES), dtype=object)
        fuertes = list(range(20))
        for i in fuertes:
            detalle["razones"][i] = "ruido_hf_fuerte"
            detalle["score"][i] = 100 - i
            detalle["z_hf"][i] = 5.0

        original = P.UMBRAL_PATRON_SISTEMICO
        try:
            P.UMBRAL_PATRON_SISTEMICO = 0.20
            self.assertIsNotNone(
                P.diagnosticar_adquisicion(detalle, NOMBRES, fuertes, [], 20.0)["patron_sistemico"]
            )
            P.UMBRAL_PATRON_SISTEMICO = 0.50
            self.assertIsNone(
                P.diagnosticar_adquisicion(detalle, NOMBRES, fuertes, [], 20.0)["patron_sistemico"]
            )
        finally:
            P.UMBRAL_PATRON_SISTEMICO = original

    def test_interferencia_respeta_la_constante_del_modulo(self):
        detalle = {k: np.zeros(len(NOMBRES)) for k in
                   ("z_std", "z_ptp", "z_hf", "z_alpha", "z_dist", "score")}
        detalle["razones"] = np.array(["normal"] * len(NOMBRES), dtype=object)

        original = P.UMBRAL_INTERFERENCIA_RED_PCT
        try:
            P.UMBRAL_INTERFERENCIA_RED_PCT = 50.0
            diag = P.diagnosticar_adquisicion(
                detalle, NOMBRES, [], [], 10.0,
                pct_ruido_red=40.0, snr_sin_red_db=15.0,
            )
            self.assertIsNone(diag["interferencia_red"])
            P.UMBRAL_INTERFERENCIA_RED_PCT = 30.0
            diag = P.diagnosticar_adquisicion(
                detalle, NOMBRES, [], [], 10.0,
                pct_ruido_red=40.0, snr_sin_red_db=15.0,
            )
            self.assertIsNotNone(diag["interferencia_red"])
        finally:
            P.UMBRAL_INTERFERENCIA_RED_PCT = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
