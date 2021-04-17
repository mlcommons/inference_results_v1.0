# Minimal dockerfile required to run the MLPerf benchmarks
# in our flow.
FROM cuda:10.2-devel-ubuntu18.04

## Install base software
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        clang-format \
        cmake \
        curl \
        gcc \
        git \
        git-lfs \
        less \
        libboost-all-dev \
        libbz2-dev \
        libcurl3-dev \
        libffi-dev \
        libgmp-dev \
        liblzma-dev \
        libssl-dev \
        libsox-dev \
        libsox-fmt-mp3 \
        locales \
        pkg-config \
        python-dev \
        python-pip \
        python-setuptools \
        python-wheel \
        python-numpy \
        python-enum34 \
        python3.7-dev \
        python3.7-distutils \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        openssh-client \
        libverilog-perl \
        ruby-full \
        rsync \
        uuid-dev \
        wget \
        vim \
        zlib1g-dev \
        zip \
        unzip \
        google-perftools \
        libgoogle-perftools-dev

## Install Bazelisk
ARG BAZELISK_VERSION=1.5.0
RUN mkdir -p /usr/local/bin
RUN curl -sS -L -o "/usr/local/bin/bazel" "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"
RUN chmod +x /usr/local/bin/bazel
COPY docker.bazelrc /etc/bazel.bazelrc

## Enable Bazel bash-completion
COPY bazel-complete.bash /etc/bash_completion.d/bazel

## Install / configure Golang
ARG GOLANG_VERSION=1.12
RUN cd /tmp && \
    wget https://dl.google.com/go/go$GOLANG_VERSION.linux-amd64.tar.gz && \
    tar -xvf go$GOLANG_VERSION.linux-amd64.tar.gz && \
    mv go /usr/local
ENV GOROOT /usr/local/go
ENV GOPATH ${HOME}/go
ENV PATH ${GOPATH}/bin:${GOROOT}/bin:${PATH}

## Install pyenv for easy Python version management
ARG PY2_VERSION=2.7.18
ARG PY3_VERSION=3.7.8
ENV PYENV_ROOT=/opt/pyenv
RUN git clone git://github.com/yyuu/pyenv.git $PYENV_ROOT
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PY2_VERSION}
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PY3_VERSION}
RUN pyenv global ${PY2_VERSION} ${PY3_VERSION}
RUN pyenv rehash

## Install PY2/PY3 requirements (supports py3.6, py3.7)
WORKDIR /app/python/requirements
ADD requirements/requirements-py2.txt requirements-py2.txt
RUN python -m pip install -r requirements-py2.txt
ADD requirements/requirements-py3.txt requirements-py3.txt
RUN python3 -m pip install -U pip setuptools
RUN python3 -m pip install -r requirements-py3.txt
RUN python3.6 -m pip install -U pip setuptools
RUN python3.6 -m pip install -r requirements-py3.txt

## Create development user that matches the current user's UID/GID
ARG UNAME=ncoresw
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${UNAME} && \
    useradd -r -u ${UID} --create-home --home-dir=/home/${UNAME} -g ${UNAME} ${UNAME} && \
    usermod -a -G ${UNAME} ${UNAME} && \
    install --directory -o ${UNAME} -g ${UNAME} -m 0755 /home/${UNAME}/.cache # Otherwise bazel generates via root
ENV HOME /home/${UNAME}

USER ${UNAME}

## Override LD_LIBRARY_PATH to prioritize Bazel TensorRT/CudDNN installs
ENV LD_LIBRARY_PATH /workspace/bazel-workspace/external/libtorch/lib:/workspace/bazel-workspace/external/libtorch_pre_cxx11_abi/lib:/workspace/bazel-workspace/external/cuda/lib:/workspace/bazel-workspace/external/cuda/lib:/workspace/bazel-workspace/external/cudnn/lib64:/workspace/bazel-workspace/external/tensorrt/lib
WORKDIR /workspace
