"""Gunicorn configuration"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "-"
accesslog = "-"
loglevel = "info"
