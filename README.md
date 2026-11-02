# WEkEO Training Notebooks (`wekeo4training`)

Welcome to the **WEkEO Training** repository! This repository contains hands-on Jupyter Notebook tutorials and training materials designed to help you access, analyze, and visualize Earth Observation (EO) data hosted on the [WEkEO DIAS](https://www.wekeo.eu/) platform.

---

## Overview

The tutorials cover various domains, including:
- Atmosphere monitoring (Copernicus Atmosphere Monitoring Service - CAMS)
- Marine environment (Copernicus Marine Service - CMEMS)
- Land monitoring (Copernicus Land Monitoring Service - CLMS)
- Climate change indicators (Copernicus Climate Change Service - C3S)
- Using the WEkEO Harmonized Data Access (HDA) API and `earthkit` workflows.

---

## Python Environment Setup

To ensure all notebooks run smoothly without dependency conflicts or missing packages, a unified Conda environment definition is provided in [`training_environment.yml`](./training_environment.yml).

This environment bundles:
- **Core Scientific Stack:** `python 3.11`, `numpy`, `pandas`, `xarray`, `matplotlib`, `pillow`
- **Geospatial & Remote Sensing Stack:** `cartopy`, `fiona`, `geopandas`, `pyproj`, `rasterio`, `rioxarray`, `shapely`
- **Jupyter Execution Tools:** `ipykernel`, `nbconvert`, `nbformat`, `nbclient`, `ipython`
- **WEkEO & EO Tools:** `hda` (Harmonized Data Access API client), `earthkit`

---

## Step-by-Step Installation Guide

Follow these steps directly inside your **JupyterHub Terminal** (or local terminal):

### 1. Clone the Repository

If you haven't cloned the repository yet, run:

```bash
git clone https://github.com/wekeo/wekeo4training.git
cd wekeo4training
```

### 2. Create the Conda Environment

Depending on your environment, pick the appropriate command:

#### Option A: JupyterHub / Shared Storage (Recommended)
On JupyterHub instances or shared NFS storage, file locks can cause SQLite errors (`database is locked`) when using the default solver. We recommend installing to a local prefix directory using the classic solver:

```bash
conda env create --solver classic -p ./trainings-python -f training_environment.yml
```

#### Option B: Standard Local Installation
If you are running on a personal machine or standard local Conda installation:

```bash
conda env create -f training_environment.yml
```

---

### 3. Activate the Environment

Activate the newly created environment:

```bash
# If installed using Option A (local prefix):
conda activate ./trainings-python

# If installed using Option B (named environment):
conda activate trainings-python
```

---

### 4. Register the Kernel in JupyterHub

Register the environment as a selectable Jupyter kernel so it appears in the JupyterLab Launcher and kernel switcher:

```bash
python -m ipykernel install --user --name trainings-python --display-name "Python (trainings-python)"
```

To verify that the kernel has been successfully installed, run:

```bash
jupyter kernelspec list
```
You should see `trainings-python` listed among the available kernels.

---

### 5. Running the Notebooks

1. **Refresh your browser** (press `F5` or reload the JupyterHub tab).
2. Open any training notebook (`.ipynb`) in the file explorer.
3. In the top-right corner of the notebook window, click on the kernel name to open the kernel selection dialog.
4. Select **`Python (trainings-python)`**.
5. You are now ready to run all cells!

---

## Managing the Environment

### Updating the Environment
If `training_environment.yml` is updated in the repository, you can update your local environment:

```bash
# For local prefix installations:
conda env update -p ./trainings-python -f training_environment.yml --prune

# For named installations:
conda env update -n trainings-python -f training_environment.yml --prune
```

### Removing the Kernel & Environment
If you need to reset or completely reinstall:

```bash
# 1. Remove the Jupyter kernel registration
jupyter kernelspec uninstall trainings-python -y

# 2. Remove the Conda environment
# For local prefix installations:
conda env remove -p ./trainings-python
rm -rf ./trainings-python

# For named installations:
conda env remove -n trainings-python -y
```

---

## Helpful Resources

- [WEkEO Documentation](https://www.wekeo.eu/docs)
- [WEkEO Harmonized Data Access (HDA) Python Client](https://pypi.org/project/hda/)
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/)