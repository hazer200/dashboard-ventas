import subprocess
import sys
import os

# Cambiar al directorio del script para que las rutas relativas funcionen
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==================== HELPERS ====================

def run_cmd(cmd, name):
    print(f"\n▶️  Ejecutando: {name}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {name} finalizado correctamente.")
        return True
    except FileNotFoundError:
        print(f"❌ Comando no encontrado: {cmd[0]}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} falló (código {e.returncode}).")
        return False

def check_files(*files):
    ok = True
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Archivo no encontrado: {f}")
            ok = False
    return ok

def run_etl():
    print("\n" + "="*50)
    print("  🗄️  PROCESO ETL")
    print("="*50)
    if not check_files("etl.py"):
        return False
    success = run_cmd([sys.executable, "etl.py"], "Proceso ETL")
    if not success:
        print("\n⛔ La ETL falló.")
    return success

def run_dashboard():
    print("\n" + "="*50)
    print("  📊  DASHBOARD STREAMLIT")
    print("="*50)
    if not check_files("app.py"):
        return False

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ventas.db")
    if not os.path.exists(db_path):
        print("⚠️  No se encontró ventas.db. Ejecuta primero la ETL.")
        resp = input("   ¿Deseas ejecutar la ETL ahora? (s/n): ").strip().lower()
        if resp == "s":
            if not run_etl():
                return False
        else:
            print("⛔ Dashboard cancelado.")
            return False

    # Buscar streamlit en el entorno virtual o en el PATH
    streamlit_cmd = os.path.join(os.path.dirname(sys.executable), "streamlit")
    if os.name == "nt":
        streamlit_cmd += ".exe"
    if not os.path.exists(streamlit_cmd):
        streamlit_cmd = "streamlit"  # fallback al PATH del sistema

    return run_cmd([streamlit_cmd, "run", "app.py"], "Streamlit Dashboard")

def run_all():
    print("\n" + "="*50)
    print("  🚀  EJECUTAR TODO (ETL + DASHBOARD)")
    print("="*50)
    if not run_etl():
        print("\n⛔ La ETL falló. No se lanzará el dashboard.")
        return False
    print("\n✨ ETL completada. Lanzando dashboard...")
    return run_dashboard()

# ==================== MENÚ ====================

def show_menu():
    print("\n" + "="*50)
    print("       🎛️  LANZADOR DEL PROYECTO")
    print("="*50)
    print("  [1]  🗄️  Solo ETL (cargar datos)")
    print("  [2]  📊  Solo Dashboard (Streamlit)")
    print("  [3]  🚀  Todo (ETL + Dashboard)")
    print("  [0]  ❌  Salir")
    print("="*50)

def main():
    # Soporte para argumento por línea de comandos: py launcher.py [1|2|3]
    if len(sys.argv) > 1:
        opcion = sys.argv[1].strip()
    else:
        show_menu()
        opcion = input("\n  Elige una opción: ").strip()

    if opcion == "1":
        run_etl()
    elif opcion == "2":
        run_dashboard()
    elif opcion == "3":
        run_all()
    elif opcion == "0":
        print("\n👋 Saliendo...")
        sys.exit(0)
    else:
        print(f"\n⚠️  Opción '{opcion}' no válida.")
        main()

if __name__ == "__main__":
    main()
