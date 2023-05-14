# ASL X-Plane 11 Simulator
Based on the popular X-Plane 11 aircraft simulator, we present a convenient Python-based simulation platform to test and benchmark the performance of perception and control algorithms when they experience Out-of-Distribution scenarios in closed-loop. This platform is refactored from the original [NASA-ULI-Xplane repo](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator/tree/main) to expand functionality, improve code quality, readability, and modularity. We offer photo-realistic graphics and accurate physics simulation Currently, the simulator offers a single control task: Vision-based autonomous taxiing. 

Users can flexibly define different OOD scenarios based on weather that may cause vision-degradation. Currently, we support variations in/addition of:
- cloud levels		
- time-of-day		
- additive image noise		
- snow		
- snowfall		
- rain 		
- motion-blur
- combinations of the above		
- several severity levels per corruption type
- linearly increasing and decreasing severity throughout an episode

### Features offered:
1. Conveniently interact with the simulator through the `XPlaneBridge` Python API, similar to the CARLA client.
2. Modular and standardized abstractions for perception and estimation to facilitate development of control/perception algorithms or use of existing systems
3. Specify and run thousands of simulations by modifying example `yaml` param files to sample environment variations
4. A single lightweight example script for sampling environments and running and recording experiments
5. Some utilities to analyze data and create videos of all episodes in an experiment
6. Track episode statistics for series of experiments anywhere through the web using [weights and biases](https://wandb.ai/)

## Setup Instructions:
1. First, follow the legacy setup instructions, available at `https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator`. Specifically, follow the legacy instructions to:
    - Download and install the X-Plane 11 simulator
    - Install the X-Camera Plugin
    - Download the 208B Grand Caravan Aircraft model
    - Download and install the X-Plane connect plugin
    - Configure the flight details in X-Plane
    - Configure the X-Camera in X-Plane
2. Clone the repository: `git clone https://github.com/StanfordASL/XPlane-ASL.git`
3. Enter the XPLANE-ASL directory: `cd XPLANE-ASL`
4. (optional) create and activate a virtual environment to install the packages:
    - to create virtual env: `python3 -m venv xplane-env --system-site-packages`
    - to activate virtual env: `source xplane-env/bin/activate`
    - to shut down virtual env: `deactivate`
    - make sure you add the name of your virtual environment to the .gitignore file! (xplane-env is already included)
5. build the aslxplane and xpc3 packages
    - ensure build package is up-to-date: `python3 -m pip install --upgrade build`
    - build packages: `python3 -m build`
6. install the image augmentation library:
    - `pip3 install imagecorruptions`
    - `pip3 install git+https://github.com/marcown/imgaug.git` (May 2023: do not use the pypi package index version, it has a bug introduced by the latest numpy version)
6. install the aslxplane and xpc3 packages: `python3 -m pip install -e ./`
7. If you create new files or packages, uninstall, rebuild, and reinstall all the packages in the reopository:
    - uninstall packages `python3 -m pip uninstall ./`
    - rebuild packages (step 4)
    - reinstall packages (step 5)

## Quick Start Workflow:
Quick-start workflow to run an experiment:
1. create a folder to store your experiment data: `mkdir your-data-directory/your-experiment-name`
2. copy the template parameter files into your experiment directory:
    - `mkdir your-data-directory/your-experiment-name/params `
    - `cp Xplane-ASL/params/simulator_params.yaml your-data-directory/your-experiment-name/params`
    - `cp Xplane-ASL/params/experiment_params.yaml your-data-directory/your-experiment-name/params`
    Note: if running data-collection to train perception models, use the `sinusoid_dataset_params.yaml` instead of `experiment_params.yaml` template
3. modify the parameter files to your desired settings
4. (optionally) to dry-run your experiment setup, initially consider running with params `debug/perception` set to `True` and `logging/log_data` set to `False` and look at the experiment runs to see if it matches desired behavior
5. enter the XPLANE-ASL directory: `cd XPlane-ASL/`
6. run your experiment by calling `trajectoryrunner.py` with your experiment directory as an argument:
    - `python3 trajectoryrunner.py relative-path-to-data-dir/your-data-directory/your-experiment-name/`

Download datasets and quickstart model checkpoints [here](https://drive.google.com/drive/folders/1n_RRKtvQqthSMuK0vbgHUMBw_FAvuwWZ?usp=sharing)

## Citation
If you use our simulator in your work, please cite its usage as:
```latex 
@article{SinhaSchmerlingPavone2023,
    title={Closing the Loop on Runtime Monitors with Fallback Safe MPC},
    author={Sinha, Rohan and Schmerling, Ed and Pavone, Marco},
    journal={arXiv preprint arXiv:tbd},
    year={2023}
}
```

## To Do:
- [ ] generate mkdocs
- [ ] figure out how to place obstacles on the runway
- [ ] change the toolbar crop pixels from a fixed value to a fraction of the monitor size such that trained xplane controllers can work for many different monitor setups...

Nice-to-haves:
- [ ] implement wrapper to conform to openai gym api specification
- [ ] implement functionality to run the xplane_bridge as a ROS node
