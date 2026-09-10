import os
import subprocess
import time
from multiprocessing import Process


def start_api(path='.'):
    subprocess.call(
        ['uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '8502'],
        cwd=path,
    )


def start_frontend(path='.'):
    subprocess.call(['streamlit', 'run', 'streamlit.py'], cwd=path)


if __name__ == '__main__':
    path = os.path.realpath(os.path.dirname(__file__))
    api = Process(target=start_api, kwargs={'path': path})
    api.start()
    frontend = Process(target=start_frontend, kwargs={'path': path})
    frontend.start()
    processes = (api, frontend)
    try:
        while all(process.is_alive() for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
