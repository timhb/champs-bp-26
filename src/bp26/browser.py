"""Headless Chromium for screenshots, PDFs and UI checks.

One-time setup: `pixi run playwright install chromium`. The cluster nodes lack some system
libraries Chromium needs; the Pixi environment provides them, so only the browser process
gets the environment's lib/ on its library path.
"""

import os
import sys
from contextlib import contextmanager

from playwright.sync_api import sync_playwright


@contextmanager
def chromium(**launch_options):
    """Yield a launched headless Chromium browser; closed on exit."""
    lib = os.path.join(sys.prefix, "lib")
    path = os.pathsep.join(filter(None, [lib, os.environ.get("LD_LIBRARY_PATH")]))
    env = {**os.environ, "LD_LIBRARY_PATH": path}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(env=env, **launch_options)
        try:
            yield browser
        finally:
            browser.close()
