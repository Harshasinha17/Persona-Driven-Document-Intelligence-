# Use prebuilt PyTorch image with CUDA already installed
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install only the extra libraries (torch already included)
RUN pip install --no-cache-dir \
    tqdm \
    numpy \
    scikit-learn \
    PyMuPDF \
    sentence-transformers

COPY . .

CMD ["python", "src/main.py"]
