# Bridging Big Data and Distributed AI Ingestion

This repository contains the coursework and interactive notebooks for the Big Data (CSCI-GA.2437) lectures on distributed data ingestion and streaming. 

## Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **uv** (for fast Python environment management)

## Setup Instructions

1. **Initialize the Virtual Environment & Install Dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Start the Local S3 (MinIO) Environment:**
   Run the MinIO server script to download the appropriate binary for your OS and start a local S3 instance on port 9000.
   ```bash
   python scripts/00_start_minio.py
   ```
   *Note: Ensure this process remains running in a background terminal during all lectures.*
   You can log in to the MinIO console at `http://127.0.0.1:9000` with the credentials `minioadmin` / `minioadmin`. Create a bucket named `cifar-streaming` via the UI.

3. **Serialize Data to Shards:**
   Pre-process and package the CIFAR-10 dataset into WebDataset Tar shards.
   ```bash
   python scripts/01_prepare_shards.py
   ```
   *Follow the printed stdout instructions to upload the shards to your MinIO bucket.*

4. **Launch Jupyter Notebooks:**
   Run the following command to begin your lectures:
   ```bash
   jupyter notebook notebooks/
   ```

## Lectures Available
- **Lecture 1:** Storage and Serialization Bridge (`Lecture_01_Storage_and_Serialization.ipynb`)
- **Lecture 2:** The Ingestion Bottleneck and Streaming (`Lecture_02_Streaming_and_Ingestion.ipynb`)
- **Lecture 3:** Distributed Paradigms (`Lecture_03_Distributed_Paradigms.ipynb`)