import platform
import urllib.request
import os
import subprocess
import stat
import time

def get_minio_url():
    system = platform.system()
    machine = platform.machine().lower()
    
    base_url = "https://dl.min.io/server/minio/release"
    
    if system == 'Darwin':
        if machine == 'arm64':
            return f"{base_url}/darwin-arm64/minio"
        else:
            return f"{base_url}/darwin-amd64/minio"
    elif system == 'Linux':
        if machine in ['arm64', 'aarch64']:
            return f"{base_url}/linux-arm64/minio"
        elif machine in ['x86_64', 'amd64']:
            return f"{base_url}/linux-amd64/minio"
    elif system == 'Windows':
        return f"{base_url}/windows-amd64/minio.exe"
    else:
        raise ValueError(f"Unsupported OS: {system}")

def download_minio():
    url = get_minio_url()
    filename = "minio.exe" if platform.system() == 'Windows' else "minio"
    
    if not os.path.exists(filename):
        print(f"Downloading MinIO from {url}...")
        urllib.request.urlretrieve(url, filename)
        if platform.system() != 'Windows':
            # chmod +x
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)
        print("Download complete.")
    else:
        print("MinIO binary already exists.")
    return filename

def start_minio(binary):
    data_dir = "./minio_data"
    os.makedirs(data_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["MINIO_ROOT_USER"] = "minioadmin"
    env["MINIO_ROOT_PASSWORD"] = "minioadmin"
    
    cmd = [f"./{binary}" if platform.system() != 'Windows' else binary, "server", data_dir, "--address", "127.0.0.1:9000"]
    
    print("Starting MinIO server on http://127.0.0.1:9000 ...")
    process = subprocess.Popen(cmd, env=env)
    
    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\nStopping MinIO...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    binary_path = download_minio()
    start_minio(binary_path)
