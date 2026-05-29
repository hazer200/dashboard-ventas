import pandas as pd
import sqlite3
import os
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio base: siempre la carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "ventas.db")

@contextmanager
def get_db_connection(db_path=None):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def run_etl(excel_path, db_path=None, chunk_size=5000):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if not os.path.exists(excel_path):
        logging.error(f"❌ Archivo Excel no encontrado: {excel_path}")
        return False

    logging.info(f"🚀 Iniciando ETL: {excel_path} -> {db_path}")
    
    # Leer Excel completo (pandas no soporta chunksize nativo para Excel)
    # Se procesará iterativamente en memoria para cumplir el requisito
    try:
        df_raw = pd.read_excel(excel_path)
        logging.info(f"📊 {len(df_raw)} registros leídos del Excel.")
    except Exception as e:
        logging.error(f"❌ Error leyendo Excel: {e}")
        return False

    # Preparar tabla
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ventas (
                Fecha TEXT, Solicitante TEXT, "Punto de Venta" INTEGER,
                Nombre TEXT, "Código Ean" INTEGER, Material INTEGER,
                Color TEXT, Tamaño TEXT, "Mat:Mundo" TEXT,
                "Mat:Grupo Articulo" TEXT, "Mat:Tipo Articulo" TEXT,
                "Mat:SubLinea" TEXT, UN INTEGER, costo REAL, Precio REAL
            )
        """)
        conn.commit()

    # Proceso iterativo por chunks
    total_chunks = (len(df_raw) + chunk_size - 1) // chunk_size
    for i in range(0, len(df_raw), chunk_size):
        chunk = df_raw.iloc[i:i+chunk_size].copy()
        
        # Transformaciones por chunk
        chunk['Fecha'] = pd.to_datetime(chunk['Fecha'], format='%d.%m.%Y', errors='coerce')
        chunk['UN'] = pd.to_numeric(chunk['UN'], errors='coerce').fillna(0).astype(int)
        chunk['costo'] = pd.to_numeric(chunk['costo'], errors='coerce').fillna(0.0)
        chunk['Precio'] = pd.to_numeric(chunk['Precio'], errors='coerce').fillna(0.0)
        
        # Guardar iterativamente
        with get_db_connection(db_path) as conn:
            chunk.to_sql('ventas', conn, if_exists='append', index=False)
        logging.info(f"✅ Chunk {i//chunk_size + 1}/{total_chunks} insertado ({len(chunk)} filas)")

    # Crear índices para optimizar consultas
    with get_db_connection(db_path) as conn:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_fecha ON ventas(Fecha)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tipo ON ventas("Mat:Tipo Articulo")')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sublinea ON ventas("Mat:SubLinea")')
        
    logging.info("✨ ETL completada exitosamente.")
    return True

if __name__ == "__main__":
    # Buscar el Excel en la carpeta actual o en la carpeta padre
    excel_name = "Base_datos_proy (1).xlsx"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, excel_name),
        os.path.join(script_dir, "Base_datos_proy.xlsx"),
        os.path.join(os.path.dirname(script_dir), excel_name),
        os.path.join(os.path.dirname(script_dir), "Base_datos_proy.xlsx"),
    ]
    excel_path = next((p for p in candidates if os.path.exists(p)), None)
    if not excel_path:
        logging.error(f"❌ No se encontró el archivo Excel en ninguna ruta conocida.")
        exit(1)
    run_etl(excel_path)