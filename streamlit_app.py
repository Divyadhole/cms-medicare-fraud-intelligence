"""
streamlit_app.py
Entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for this file at repo root.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import *  # noqa
