# Use a lightweight CUDA runtime base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV TZ=Europe/Madrid
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/torch/
ENV VENV_PATH="/venv"
ENV MALLET_VERSION=202108
ENV MALLET_URL="https://github.com/mimno/Mallet/releases/download/v${MALLET_VERSION}/Mallet-${MALLET_VERSION}-bin.tar.gz"
ENV MALLET_HOME=/opt/mallet
ENV PATH="$MALLET_HOME/bin:$PATH"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# Install system dependencies and ensure Java 8 compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    poppler-utils \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git ant wget curl \
    libbz2-dev libffi-dev libgdbm-dev libncurses5-dev \
    libnss3-dev libreadline-dev libsqlite3-dev libssl-dev \
    tzdata zlib1g-dev libgl1-mesa-glx libgl1-mesa-dev libpython3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/*

# Set Java 8 as the default
RUN update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

# Verify Java version (should be Java 8)
RUN java -version

# Create and activate a Python virtual environment
RUN python3.10 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip cache purge

# Set working directory
WORKDIR /np-tools

# Copy only requirements file first to optimize caching
COPY requirements.txt .

# Install project dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Install spaCy models explicitly in the virtual environment
RUN pip install --no-cache-dir spacy
RUN python -m spacy download es_dep_news_trf
RUN python -m spacy download en_core_web_trf

# Download and install Mallet (removing unnecessary files to save space)
RUN wget $MALLET_URL -O /tmp/mallet.tar.gz && \
    tar -xvzf /tmp/mallet.tar.gz -C /opt && \
    mv /opt/Mallet-${MALLET_VERSION} /opt/mallet && \
    rm /tmp/mallet.tar.gz && \
    rm -rf /opt/mallet/sample-data  # Remove sample data to reduce image size
#COPY mallet-2.0.8/ /opt/mallet/

# Download and cache the sentence transformer model
ARG MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${MODEL_NAME}').save('/tmp/huggingface')"

RUN mkdir -p /venv/lib/python3.12/site-packages/frontend/static


# Copy project files
COPY . .

# Initialize sentence transformer
RUN python initialize_st.py

# Copy environment file
COPY .env .

# Expose application port
EXPOSE 102

# Run the application inside the virtual environment
CMD ["python", "app.py"]