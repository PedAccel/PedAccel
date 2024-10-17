# PedAccel
JHU BME Design Team Data Analysis Pipeline and Clinical Study Packages


# Installation
We advocate for `micromamba` as the package manager. To install `micromamba`, refer to the [installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Alternatively, you can use the `conda` environment management package, which should be installed by default on most research computers and clusters. To verify that you have a `conda` installations, try `conda help`.

To install `PedAccel`:
```
git clone https://github.com/fyng/PedAccel.git
cd PedAccel

# if using micromamba
micromamba env create -f environment.yaml
# if using conda
conda env create -f environment.yaml
```

To run the data preprocessing:
```
cd data_analysis/PythonPipeline/preprocessing.py

# change the data folder path according to the code comments in the file

# if using micromamba
micromamba activate pedaccel
# if using conda
conda activate pedaccel 

python preprocessing.py
```

