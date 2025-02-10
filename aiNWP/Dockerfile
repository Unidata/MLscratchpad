# Heavily borrowed from docker-stacks/minimal-notebook/
# https://github.com/jupyter/docker-stacks/blob/main/minimal-notebook/Dockerfile

ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

ENV DEFAULT_ENV_NAME=tm-fall-2024 EDITOR=nano VISUAL=nano

LABEL maintainer="Unidata <support-gateway@unidata.ucar.edu>"

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim nano curl zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $NB_UID

ADD environment.yml /tmp

RUN mamba install --quiet --yes \
      'conda-forge::nb_conda_kernels' \
      'conda-forge::jupyterlab-git' \
      'conda-forge::ipywidgets' && \
    mamba env update --name $DEFAULT_ENV_NAME -f /tmp/environment.yml && \
    pip install --no-cache-dir nbgitpuller && \
    mamba clean --all -f -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    rm -rf /home/$NB_USER/.node-gyp && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install the cuda toolkit, which includes the nvidia cuda compiler (nvcc), used
# to compile the "apex" package necessary for some portions of the earth2mip
# code: https://github.com/NVIDIA/apex
RUN wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run && \
    sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit && \
    rm /tmp/cuda_12.1.0_530.30.02_linux.run  && \
    # Install the apex package in the default environment
    cd /tmp && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    conda run -n $DEFAULT_ENV_NAME \
        pip install -v --disable-pip-version-check --no-cache-dir \
        --no-build-isolation --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./

COPY GPU_sanity_check.ipynb Acknowledgements.ipynb \
    default_kernel.py .condarc /

ARG JUPYTER_SETTINGS_DIR=/opt/conda/share/jupyter/lab/settings/
COPY overrides.json $JUPYTER_SETTINGS_DIR

USER $NB_UID
