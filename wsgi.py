"""
Production WSGI entry point.

Usage:
    waitress-serve --listen=0.0.0.0:5000 wsgi:app
    # or
    python wsgi.py
"""

from app import app

if __name__ == "__main__":
    try:
        from waitress import serve
        print("Starting NeuroMap on http://127.0.0.1:5000 (Waitress WSGI)")
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        print("Waitress not installed; starting NeuroMap on http://127.0.0.1:5000 (Flask WSGI)")
        app.run(host="0.0.0.0", port=5000, debug=False)

