"""Lightweight web GUI (pywebview + FastAPI) — the shipped desktop UI.

Replaces the PySide6/Qt screens with a system-WebView window driven by a local
FastAPI server, keeping the distribution light (~10 MB of UI deps vs ~1.2 GB Qt).
The dashboard and the review screen are plain HTML/CSS/JS under :mod:`static`.
"""
