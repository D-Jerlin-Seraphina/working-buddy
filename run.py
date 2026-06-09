#!/usr/bin/env python3
"""
Employee Retention Buddy — Main Entry Point
Usage:
    python run.py api          # Start FastAPI backend
    python run.py dashboard    # Start Streamlit dashboard
    python run.py setup        # Initialize DB + generate synthetic data
    python run.py all          # Start both (API in background, dashboard in foreground)
"""
import sys
import os
import subprocess
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_api():
    print("[+] Starting FastAPI backend on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--timeout-keep-alive", "300",
        "--timeout-graceful-shutdown", "300"
    ], cwd=ROOT)


def run_dashboard():
    print("[+] Starting Streamlit dashboard on http://localhost:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", "8501"
    ], cwd=ROOT)


def run_setup(n_employees: int = 50):
    print("[+] Initializing database...\n")
    from backend.utils.database import init_db
    init_db()
    print("  [OK] Database tables created")
    subprocess.run([
        sys.executable, "-m", "data.generate_synthetic",
        "--employees", str(n_employees)
    ], cwd=ROOT)


def run_all():
    import threading
    import time
    t = threading.Thread(target=run_api, daemon=True)
    t.start()
    time.sleep(3)
    run_dashboard()


def main():
    parser = argparse.ArgumentParser(description="Employee Retention Buddy")
    parser.add_argument("command", choices=["api", "dashboard", "setup", "all"],
                        nargs="?", default="api")
    parser.add_argument("--employees", type=int, default=50)
    args = parser.parse_args()

    {"api": run_api, "dashboard": run_dashboard,
     "setup": lambda: run_setup(args.employees), "all": run_all}[args.command]()


if __name__ == "__main__":
    main()
