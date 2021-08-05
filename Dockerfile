# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV CONDA_ENV_NAME=myenv
ENV PYTHON_VERSION=3.8

RUN mkdir /workdir
RUN apt-get update && \
     apt purge -y python3 python3-dev && \
     apt install -y python3-pip python3.8 python3.8-dev libsm6 libxext6 libxrender-dev gnuplot git

RUN python3.8 -m pip install pip

RUN python3.8 -m pip install --upgrade pip setuptools
RUN python3.8 -m pip install pipenv

WORKDIR /workdir
COPY Pipfile* /workdir/
RUN rm -rf /usr/local/lib/python3.8/dist-packages/numpy*
RUN pip install --force-reinstall --no-deps numpy

ENV LANG C.UTF-8
ENV WANDB_API_KEY "4631776b948ff1b794c99e015259b0812df58e59"

RUN pipenv install --system --deploy --skip-lock

COPY ./configs /workdir/configs
COPY *.py /workdir/
COPY ./src /workdir/src

# Set ${CONDA_ENV_NAME} to default virutal environment
#RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
