import gc
import os

bind = "localhost:23847"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
preload_app = True
timeout = 200
max_requests = 500
max_requests_jitter = 10

accesslog = "-"
errorlog = "-"


def pre_fork(server, worker):  # noqa: ARG001
    """Freeze preloaded objects so the forked worker keeps sharing their memory."""
    gc.freeze()
