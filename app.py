"""
EVolve: Intelligent EV Route Charging Optimizer
Root deployment entrypoint (redirects to dashboard/app.py)
Supports Streamlit Community Cloud, Docker, Render, Hugging Face Spaces, and Railway.
"""

import os
import sys
import runpy

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Path to the primary Streamlit dashboard application
DASHBOARD_APP_PATH = os.path.join(ROOT_DIR, "dashboard", "app.py")

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(DASHBOARD_APP_PATH, run_name="__main__")
