# WEkEO Training Notebooks (`wekeo4training`)

Welcome to the official repository for **WEkEO training sessions and webinars**!

This repository contains Jupyter Notebooks developed and presented during WEkEO training events. They provide hands-on workflows demonstrating how to search, access, subset, and visualize Copernicus Earth Observation (EO) data using WEkEO computing resources and APIs.

* **Upcoming & Past Events:** Find agendas, presentation slides, and registration details on [events.wekeo.eu](https://events.wekeo.eu/).
* **Webinar Replays & Video Tutorials:** Watch the recorded sessions, demos, and walkthroughs on the [WEkEO YouTube Channel (@wekeodias6693)](https://www.youtube.com/@wekeodias6693).

---

## Overview

The tutorials cover practical applications across Copernicus service domains:

* **Atmosphere:** Copernicus Atmosphere Monitoring Service (CAMS)
* **Marine:** Copernicus Marine Service (CMEMS)
* **Land:** Copernicus Land Monitoring Service (CLMS)
* **Climate:** Copernicus Climate Change Service (C3S)
* **Data Access Workflows:** Harmonized Data Access (HDA) API and `earthkit` tools.

---

## Python Environment Setup

To ensure all notebooks across this repository execute smoothly without missing dependencies or package version conflicts, a unified Conda environment definition is provided in [`training_environment.yml`](./training_environment.yml).

### Stack Highlights

* **Core Scientific Stack:** `python 3.11`, `numpy`, `pandas`, `xarray`, `matplotlib`, `pillow`
* **Geospatial & Cartography:** `cartopy`, `fiona`, `geopandas`, `pyproj`, `rasterio`, `rioxarray`, `shapely`
* **Jupyter Ecosystem:** `ipykernel`, `nbconvert`, `nbformat`, `nbclient`, `ipython`
* **Data Access Clients:** `hda` (WEkEO Harmonized Data Access API), `earthkit`

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

Choose the installation option suited to your setup:

#### Option A: JupyterHub / Shared Storage (Recommended)

On JupyterHub sessions or shared NFS filesystems, SQLite file locks can cause errors (`database is locked`) when using the default solver. We recommend installing to a local prefix folder using the classic solver:

```bash
conda env create --solver classic -p ./trainings-python -f training_environment.yml
```

#### Option B: Standard Local Workstation

If you are running on a local desktop or standard Conda installation:

```bash
conda env create -f training_environment.yml
```

### 3. Activate the Environment

Activate the newly created environment:

```bash
# If installed via Option A (local prefix):
conda activate ./trainings-python

# If installed via Option B (named environment):
conda activate trainings-python
```

### 4. Register the Kernel in JupyterHub

Because `ipykernel` is bundled directly within the environment file, you can register it as an accessible Jupyter kernel with a single command:

```bash
python -m ipykernel install --user --name trainings-python --display-name "Python (trainings-python)"
```

To verify that the kernel is registered:

```bash
jupyter kernelspec list
```

You should see `trainings-python` in the list of available kernels.

### 5. Running the Notebooks

1. **Refresh your browser** (press `F5` or click reload on your JupyterLab tab).
2. Browse through the repository and open any training notebook (`.ipynb`).
3. Click the current kernel name in the top-right corner of the notebook toolbar to open the kernel selector.
4. Select **`Python (trainings-python)`**.
5. All cells are now ready to run!

---

## Environment Management

### Updating Dependencies

If `training_environment.yml` receives updates from the repository:

```bash
# For local prefix installations:
conda env update -p ./trainings-python -f training_environment.yml --prune

# For named installations:
conda env update -n trainings-python -f training_environment.yml --prune
```

### Removing the Kernel & Environment

To clean up or perform a fresh reinstall:

```bash
# 1. Remove the registered Jupyter kernel
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

* **WEkEO Training Portal:** [events.wekeo.eu](https://events.wekeo.eu/)
* **WEkEO Video Tutorials & Replays:** [YouTube (@wekeodias6693)](https://www.youtube.com/@wekeodias6693)
* **WEkEO Documentation & Knowledge Base:** [wekeo.eu/docs](https://www.wekeo.eu/docs)
* **Harmonized Data Access (HDA) Python Client:** [pypi.org/project/hda](https://pypi.org/project/hda/)