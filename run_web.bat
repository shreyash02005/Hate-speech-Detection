@echo off
echo Starting Hate Speech Detection Web App...
echo Open http://127.0.0.1:8000 in your browser
echo Press Ctrl+C to stop the server
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py"
