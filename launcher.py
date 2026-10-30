"""Lamplight desktop app: starts the private local server and shows it in its own window (no browser).

First run: a friendly setup screen installs Ollama (the engine that runs the AI) if it is missing and downloads
the AI models. Everything the app saves goes to %LOCALAPPDATA%\\Lamplight, so updating or reinstalling keeps it.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

APP_NAME = "Lamplight"
PREFERRED_PORT = 18765          # a fixed port keeps the window's saved settings (chosen document, mode) between runs
OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def base_dir():
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def data_dir():
    root = os.environ.get("LAMPLIGHT_HOME") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), APP_NAME)
    os.makedirs(root, exist_ok=True)
    return root


def quiet_streams(folder):
    """A windowed app has no console. Send anything the program prints to a log file instead of crashing."""
    try:
        log = open(os.path.join(folder, "lamplight.log"), "a", encoding="utf-8", buffering=1)
        if sys.stdout is None or getattr(sys, "frozen", False):
            sys.stdout = log
        if sys.stderr is None or getattr(sys, "frozen", False):
            sys.stderr = log
    except OSError:
        pass


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def pick_port():
    if port_free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_up(port, seconds=40):
    end = time.time() + seconds
    while time.time() < end:
        if not port_free(port):
            return True
        time.sleep(0.2)
    return False


class Engine:
    """Everything about Ollama and the models. Kept apart from the window so it can be tested."""

    def __init__(self, server):
        self.server = server
        self.models = [server.MODELS["local"], server.EMBED_MODEL]

    def ollama_exe(self):
        found = shutil.which("ollama")
        if found:
            return found
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                p = os.path.join(root, "Programs", "Ollama", "ollama.exe") if root == os.environ.get("LOCALAPPDATA") else os.path.join(root, "Ollama", "ollama.exe")
                if os.path.exists(p):
                    return p
        return None

    def running(self):
        try:
            self.server.ollama.list()
            return True
        except Exception:
            return False

    def installed_models(self):
        try:
            names = {m.model for m in self.server.ollama.list().models}
        except Exception:
            return set()
        return {n for n in names if n}

    def has_model(self, wanted, have):
        return any(h == wanted or h.split(":")[0] == wanted.split(":")[0] and ":" not in wanted for h in have)

    def status(self):
        running = self.running()
        have = self.installed_models() if running else set()
        return {
            "installed": bool(self.ollama_exe()) or running,
            "running": running,
            "models": [{"name": m, "ready": self.has_model(m, have)} for m in self.models],
        }

    def start_ollama(self, wait=25):
        exe = self.ollama_exe()
        if not exe:
            return False
        try:
            subprocess.Popen([exe, "serve"], creationflags=NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        except OSError:
            return False
        end = time.time() + wait
        while time.time() < end:
            if self.running():
                return True
            time.sleep(0.5)
        return False

    def download_installer(self, progress):
        target = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
        req = urllib.request.Request(OLLAMA_URL, headers={"User-Agent": "Lamplight"})
        with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                progress(got, total)
        return target

    def install_ollama(self, progress):
        """Downloads the official installer from ollama.com over HTTPS and runs it quietly. Returns None, or an error message."""
        if os.name != "nt":
            return "Automatic install is only for Windows. Install Ollama from ollama.com, then reopen Lamplight."
        try:
            path = self.download_installer(progress)
        except Exception as e:
            print("ollama download failed:", repr(e))
            return "Could not download Ollama. Check your internet connection and try again."
        try:
            code = subprocess.run([path, "/SILENT"], creationflags=NO_WINDOW, timeout=900).returncode
        except Exception as e:
            print("ollama install failed:", repr(e))
            return "The Ollama installer could not run. Install it yourself from ollama.com, then reopen Lamplight."
        if code != 0:
            return "The Ollama installer stopped (code %s). Install it yourself from ollama.com, then reopen Lamplight." % code
        if not self.running() and not self.start_ollama():
            return "Ollama installed but did not start. Restart your computer and reopen Lamplight."
        return None

    def pull(self, model, progress):
        """Download one model. progress(done_bytes, total_bytes, status). Returns None or an error message."""
        try:
            for part in self.server.ollama.pull(model, stream=True):
                total = getattr(part, "total", None) or 0
                done = getattr(part, "completed", None) or 0
                progress(done, total, getattr(part, "status", "") or "")
        except Exception as e:
            print("model download failed:", model, repr(e))
            return "Could not download %s. Check your internet connection and try again." % model
        return None


class Api:
    """What the setup screen can ask for. Each call returns quickly; slow work reports through setup.* callbacks."""

    def __init__(self, engine, window_ref, app_url):
        self.engine, self.window_ref, self.app_url = engine, window_ref, app_url
        self.busy = False

    def _js(self, code):
        w = self.window_ref()
        if w is not None:
            try:
                w.evaluate_js(code)
            except Exception as e:
                print("js error:", repr(e))

    def status(self):
        return self.engine.status()

    def open_app(self):
        w = self.window_ref()
        if w is not None:
            w.load_url(self.app_url)

    def start(self):
        if self.busy:
            return False
        self.busy = True
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        import json
        try:
            st = self.engine.status()
            if not st["installed"]:
                self._js("setup.step('ollama','working','Downloading Ollama (the engine that runs the AI)')")
                err = self.engine.install_ollama(lambda d, t: self._js("setup.bar('ollama',%d,%d)" % (d, t)))
                if err:
                    self._js("setup.fail(%s)" % json.dumps(err))
                    return
            elif not st["running"]:
                self._js("setup.step('ollama','working','Starting Ollama')")
                if not self.engine.start_ollama():
                    self._js("setup.fail(%s)" % json.dumps("Ollama is installed but did not start. Restart your computer and reopen Lamplight."))
                    return
            self._js("setup.step('ollama','done','Ollama is ready')")
            have = self.engine.installed_models()
            for i, name in enumerate(self.engine.models):
                key = "model%d" % i
                if self.engine.has_model(name, have):
                    self._js("setup.step(%s,'done','Ready')" % json.dumps(key))
                    continue
                self._js("setup.step(%s,'working','Downloading')" % json.dumps(key))
                err = self.engine.pull(name, lambda d, t, s, k=key: self._js("setup.bar(%s,%d,%d)" % (json.dumps(k), d, t)))
                if err:
                    self._js("setup.fail(%s)" % json.dumps(err))
                    return
                self._js("setup.step(%s,'done','Ready')" % json.dumps(key))
            self._js("setup.finished()")
        finally:
            self.busy = False


def start_server(port):
    import uvicorn
    import server
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True).start()
    return server, srv


def main(webview=None):
    folder = data_dir()
    quiet_streams(folder)
    os.chdir(folder)                                   # the app keeps docs, chats and facts next to here
    sys.path.insert(0, base_dir())
    for var in ("STUDY_DESK_MODE", "STUDY_DESK_HOSTS", "STUDY_DESK_ALLOW_HTTP"):
        os.environ.pop(var, None)                      # the desktop app is always the private, single-person mode
    os.environ.setdefault("STUDY_DESK_WHISPER_DIR", os.path.join(folder, "speech-models"))   # the speech model downloads here once, and stays
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    port = pick_port()
    server, _srv = start_server(port)
    if not wait_until_up(port):
        raise SystemExit("Lamplight could not start its local engine.")
    url = "http://127.0.0.1:%d/" % port
    engine = Engine(server)
    if webview is None:
        import webview as webview
    holder = {}
    api = Api(engine, lambda: holder.get("window"), url)
    ready = all(m["ready"] for m in engine.status()["models"]) and engine.running()
    first = url if ready else "file:///" + os.path.join(base_dir(), "setup.html").replace("\\", "/")
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True    # lets "Download as Word file" save from inside the app
    except Exception:
        pass
    window = webview.create_window(APP_NAME, first, js_api=api, width=1180, height=800, min_size=(360, 520), background_color="#120e0a")
    holder["window"] = window
    webview.start(storage_path=os.path.join(folder, "webview"), private_mode=False)


if __name__ == "__main__":
    main()
