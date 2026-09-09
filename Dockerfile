FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/opt/nltk_data

RUN sed -i \
        -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python --version

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements-project.txt /app/requirements-project.txt
RUN python -m pip install --no-build-isolation -r requirements-project.txt

COPY sam3 /app/sam3
RUN python -m pip install /app/sam3 \
    && python -m pip install git+https://github.com/openai/CLIP.git \
    && NLTK_ALLOW_PROXIED_URLOPEN=1 \
       python -m nltk.downloader -d /opt/nltk_data wordnet

COPY . /app

RUN python -m pip install setuptools==80.9.0 \
    && python -c "import torch, torchvision, transformers, cv2, nltk, clip, sam3"

EXPOSE 8888

CMD ["python", "-m", "uvicorn", "viewer:app", "--host", "0.0.0.0", "--port", "8888"]
