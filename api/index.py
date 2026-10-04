"""
Vercel Serverless Entrypoint for Flask.
Exposes the WSGI `app` callable with PATH_INFO normalization for Vercel's Python runtime.
"""

import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app


class VercelPathMiddleware:
    """
    Middleware that restores the user's original requested URL path
    from Vercel's HTTP_X_FORWARDED_URI header so Flask routes match accurately.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Prefer Vercel's true client requested URI if available
        forwarded_uri = environ.get("HTTP_X_FORWARDED_URI") or environ.get("HTTP_X_MATCHED_PATH")
        if forwarded_uri:
            clean_path = forwarded_uri.split("?")[0]
            environ["PATH_INFO"] = clean_path
        else:
            # 2. Normalize path if Vercel passes internal script path
            path = environ.get("PATH_INFO", "")
            if path.startswith("/api/index.py"):
                remainder = path[len("/api/index.py") :]
                environ["PATH_INFO"] = remainder if remainder else "/"
            elif path.startswith("/api/index"):
                remainder = path[len("/api/index") :]
                environ["PATH_INFO"] = remainder if remainder else "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with the Vercel path normalizer
app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
