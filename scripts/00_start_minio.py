"""
00_start_minio.py — Local S3 Object Storage Server Launcher
============================================================

PURPOSE:
    This script automates the download and execution of MinIO, an open-source
    object storage server that is API-compatible with Amazon S3. In production,
    teams use AWS S3, Google Cloud Storage, or Azure Blob Storage. For local
    development and teaching, MinIO provides an identical API surface without
    any cloud dependency or cost.

BACKGROUND — WHY OBJECT STORAGE?
    Traditional filesystems (ext4, NTFS, APFS) organize data hierarchically
    using directories and inodes. Object storage (S3) is fundamentally different:
    - Data is stored as "objects" inside flat "buckets" (no nested directories).
    - Each object is identified by a unique key (string), not a file path.
    - Objects are accessed over HTTP (GET/PUT/DELETE), making them network-native.
    - This design scales horizontally: adding more disks or nodes linearly
      increases capacity without restructuring the namespace.

    For AI/ML workloads, object storage is critical because:
    1. Training data (often terabytes) lives remotely in the cloud.
    2. GPUs stream data over the network rather than from local SSDs.
    3. The S3 API is the universal standard across all cloud providers.

USAGE:
    python scripts/00_start_minio.py
    Then open http://127.0.0.1:9000 in your browser.
    Credentials: minioadmin / minioadmin
"""

import platform       # Detects the host OS (Darwin=macOS, Linux, Windows)
import urllib.request # Standard library HTTP client for downloading files
import os             # File system operations (path checks, directory creation)
import subprocess     # Launches external processes (the MinIO server binary)
import stat           # File permission constants (for chmod +x on Unix)
import time           # Sleep loop to keep the script alive while MinIO runs


def get_minio_url():
    """
    Determines the correct MinIO binary download URL based on the host system.

    MinIO distributes pre-compiled binaries for each OS/architecture combination.
    Modern Apple Silicon Macs use ARM64 ("arm64"), while older Macs and most
    Linux servers use x86_64 ("amd64"). This function maps the Python
    `platform.system()` and `platform.machine()` values to the correct URL.

    Returns:
        str: The download URL for the MinIO server binary.

    Raises:
        ValueError: If the OS is not recognized (e.g., FreeBSD).
    """
    system = platform.system()       # 'Darwin' (macOS), 'Linux', or 'Windows'
    machine = platform.machine().lower()  # 'arm64', 'x86_64', 'amd64', etc.
    
    # MinIO's official release CDN endpoint
    base_url = "https://dl.min.io/server/minio/release"
    
    if system == 'Darwin':  # macOS
        if machine == 'arm64':
            # Apple Silicon (M1/M2/M3/M4 chips) — ARM64 architecture
            return f"{base_url}/darwin-arm64/minio"
        else:
            # Intel Macs — x86_64 architecture
            return f"{base_url}/darwin-amd64/minio"
    elif system == 'Linux':
        if machine in ['arm64', 'aarch64']:
            # ARM-based Linux (e.g., AWS Graviton, Raspberry Pi)
            return f"{base_url}/linux-arm64/minio"
        elif machine in ['x86_64', 'amd64']:
            # Standard x86 Linux servers (most common in data centers)
            return f"{base_url}/linux-amd64/minio"
    elif system == 'Windows':
        # Windows only supports x86_64 for MinIO
        return f"{base_url}/windows-amd64/minio.exe"
    else:
        raise ValueError(f"Unsupported OS: {system}")


def download_minio():
    """
    Downloads the MinIO binary if it doesn't already exist in the current directory.

    On Unix systems (macOS/Linux), the binary must be marked as executable
    using chmod. The `stat.S_IEXEC` flag sets the owner-execute bit,
    equivalent to running `chmod +x minio` in the terminal.

    Returns:
        str: The filename of the downloaded (or existing) binary.
    """
    url = get_minio_url()
    # On Windows, executables require the .exe extension
    filename = "minio.exe" if platform.system() == 'Windows' else "minio"
    
    if not os.path.exists(filename):
        print(f"Downloading MinIO from {url}...")
        # urlretrieve performs a blocking HTTP GET and saves to disk
        urllib.request.urlretrieve(url, filename)
        if platform.system() != 'Windows':
            # Make the binary executable (equivalent to: chmod +x minio)
            # stat.S_IEXEC = 0o100 = owner execute permission bit
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)
        print("Download complete.")
    else:
        print("MinIO binary already exists.")
    return filename


def start_minio(binary):
    """
    Launches the MinIO server as a child process.

    CONFIGURATION:
    - Data directory: ./minio_data (all bucket data is stored here on disk)
    - Bind address: 127.0.0.1:9000 (localhost only, not exposed to network)
    - Credentials: Set via environment variables MINIO_ROOT_USER and
      MINIO_ROOT_PASSWORD. In production, these would be strong secrets
      managed by a vault service (e.g., AWS Secrets Manager, HashiCorp Vault).

    IMPORTANT CONCEPTS:
    - subprocess.Popen() launches the process WITHOUT blocking Python.
      This is different from subprocess.run() which waits for completion.
    - We copy os.environ to inherit the current PATH and system variables,
      then inject our MinIO-specific credentials into the copy.
    - The while loop keeps this Python script alive. When you press Ctrl+C,
      KeyboardInterrupt is caught and we gracefully terminate MinIO.

    Args:
        binary (str): Path to the MinIO executable.
    """
    # MinIO stores all object data in this local directory
    # Each bucket becomes a subdirectory, each object becomes a file
    data_dir = "./minio_data"
    os.makedirs(data_dir, exist_ok=True)  # Create if it doesn't exist
    
    # Environment variables configure MinIO's authentication
    # These are the credentials you'll use in boto3, the web console, etc.
    env = os.environ.copy()  # Inherit system PATH, locale, etc.
    env["MINIO_ROOT_USER"] = "minioadmin"
    env["MINIO_ROOT_PASSWORD"] = "minioadmin"
    
    # Construct the command: ./minio server ./minio_data --address 127.0.0.1:9000
    # --address binds to localhost only (safe for development)
    cmd = [
        f"./{binary}" if platform.system() != 'Windows' else binary,
        "server",           # MinIO subcommand to start the storage server
        data_dir,           # Where to store bucket/object data on disk
        "--address",        # Flag to specify the listening address
        "127.0.0.1:9000"    # Bind to localhost port 9000
    ]
    
    print("Starting MinIO server on http://127.0.0.1:9000 ...")
    # Popen launches the process in the background (non-blocking)
    process = subprocess.Popen(cmd, env=env)
    
    try:
        # Keep the script alive while MinIO runs
        # process.poll() returns None if the process is still running,
        # or the exit code if it has terminated
        while True:
            time.sleep(1)
            if process.poll() is not None:
                # MinIO exited on its own (error or manual shutdown)
                break
    except KeyboardInterrupt:
        # Ctrl+C sends SIGINT to Python, caught here
        print("\nStopping MinIO...")
        process.terminate()  # Send SIGTERM to MinIO for graceful shutdown
        process.wait()       # Block until MinIO actually exits


# ============================================================================
# ENTRY POINT
# ============================================================================
# The `if __name__ == "__main__"` guard ensures this code only runs when the
# script is executed directly (python scripts/00_start_minio.py), not when
# it is imported as a module by another Python file.
if __name__ == "__main__":
    binary_path = download_minio()
    start_minio(binary_path)
