

import os
from pypdf import PdfReader, PdfWriter


def combinar_informes_pdf(rutas_pdf, ruta_salida, logger=None):
    
    def log(msg):
        if logger:
            try:
                logger(msg)
            except Exception:
                pass

    writer = PdfWriter()
    incluidos = []
    omitidos = []

    for ruta in rutas_pdf:
        if not ruta or not os.path.exists(ruta):
            omitidos.append(ruta)
            log(f"[Aviso] No encontré el PDF: {ruta}")
            continue
        try:
            reader = PdfReader(ruta)
            for pagina in reader.pages:
                writer.add_page(pagina)
            incluidos.append(ruta)
            log(f"[OK] Agregado al consolidado: {os.path.basename(ruta)}")
        except Exception as e:
            omitidos.append(ruta)
            log(f"[Aviso] No pude leer '{ruta}': {e}")

    if not incluidos:
        raise ValueError("Ninguno de los PDF seleccionados se pudo leer. No se generó el consolidado.")

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)) or ".", exist_ok=True)
    with open(ruta_salida, "wb") as f:
        writer.write(f)

    log(f"[OK] PDF consolidado guardado en: {ruta_salida}")
    return ruta_salida, incluidos, omitidos