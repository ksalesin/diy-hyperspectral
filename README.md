# DIY hyperspectral imaging via polarization-induced spectral filters

This repo provides code and notebooks for convenience in reproducing the method of "DIY hyperspectral imaging via polarization-induced spectral filters." Katherine Salesin, Dario Seyb, Sarah Friday, and Wojciech Jarosz. Proceedings of ICCP. 2022.

## Setting up your environment

Set up and activate the required python packages in a new conda environment by running:
```
conda env create --file environment.yml
conda activate hyperspectral
```
You will also need MATLAB and the included MATLAB Engine API for Python in order to run the solver with constraints. See [Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for instructions on how to install the Python API locally.

If you wish to install the required python packages manually, this is the minimal list of required packages and tested versions:

| Package      | Version |
| ------------ | ------- |
| numpy        | 1.19.2  |
| notebook     | 6.4.3   |
| ipywidgets   | 7.6.5   |
| matplotlib   | 3.3.4   |
| scipy        | 1.5.2   |
| colormath    | 3.0.0   |
| tqdm         | 4.63.0  |
| scikit-learn | 0.24.2  |
| plotly       | 5.6.0   |
| rawpy        | 0.16.0  |

## File guide

The Jupyter notebooks in `notebooks/` provide sample code for the algorithms and calibration procedures presented in our paper. You will not be able to use the interactive elements (sliders, plotly graphs, etc.) using the notebook viewers in Visual Studio Code or Github; the notebooks must be downloaded and run locally.
In the terminal, run:
```
cd notebooks
jupyter notebook
```
to start the notebooks. We recommend starting with the notebook `filter_playground` to play with the transmission spectra of our filters!

* **birefringence_calibration_real_world_plots**: load real-world birefringence calibration images and visualize the relationship between rotation of the analyzer and measured intensity
* **birefringence_calibration_sim**: use the method of [\[Belendez et al. 2009\]](https://physlab.lums.edu.pk/images/3/30/Cellophane2.pdf) to solve for birefringence under ideal conditions using simulated data
* **birefringence_calibration**: load real-world birefringence calibration images and use the method of [\[Belendez et al. 2009\]](https://physlab.lums.edu.pk/images/3/30/Cellophane2.pdf) to solve for birefringence
* **choose_optimal_filters**: our algorithm which chooses a discrete set of filters from a continuous space
* **filter_playground**: play with the transmission spectra of our filters!
* **recover_real_world_color_checker**: load real-world ColorChecker images, run white balance to recover an estimate of lighting and sensor response spectra, and reconstruct the reflectance spectrum of each ColorChecker squares
* **recover_real_world_scene**: load real-world images, run white balance to recover an estimate of lighting and sensor response spectra, and reconstruct the reflectance spectrum of each pixel