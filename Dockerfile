# cuda<=11.5 for RAPIDS, cuda <=11.3 for PyTorch
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV HOME /root
WORKDIR "$HOME"

# ------------ #
# Localization #
# ------------ #

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
ENV TZ=Asia/Tokyo

# ------ #
# Python #
# ------ #

RUN apt-get install -y python3.8 python-is-python3 python3-pip
RUN pip install --upgrade pip
# for matplotlib interface
RUN apt-get install -y python3-tk tk-dev

# --- #
# pip #
# --- #

COPY requirements.txt "$HOME"
RUN pip install -r requirements.txt

CMD ["/bin/bash"]