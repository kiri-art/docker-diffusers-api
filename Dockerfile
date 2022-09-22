# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Dev: docker build --build-arg HF_AUTH_TOKEN=${HF_AUTH_TOKEN} ...
# Banana: currently, comment out ARG and set by hand ENV line.
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# Which model to download and use; fork / downstream specific.
ADD DOWNLOAD_VARS.py .

# Add your model weight files 
# (in this case we have a python script)
ADD loadModel.py .
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD send.py .
ADD app.py .

# Runtime vars (for init and inference); fork / downstream specific.
ADD APP_VARS.py .

CMD python3 -u server.py
