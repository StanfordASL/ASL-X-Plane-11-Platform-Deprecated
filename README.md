The ASL's port of the NASA-ULI-Xplane repository. Refactored the ULI-Xplane repo to improve code quality, readability, and functionality

## To Do:
- [ ] generate mkdocs
- [ ] generate mkdocs
- [ ] reimplement the sinusoid controller for data generation
- [ ] reimplement the data logger
- [ ] improve code quality of the estimation stack
- [ ] regress new taxinet model, the heading error estimates are too bad on this one
Nice-to-haves:
- [ ] implement wrapper to conform to openai gym api specification
- [ ] implement functionality to run the xplane_bridge as a ROS node


## Setup Instructions:
1. Clone the repository: `git clone https://github.com/StanfordASL/XPlane-ASL.git`
2. Enter the XPLANE-ASL directory: `cd XPLANE-ASL`
3. (optional) create and activate a virtual environment to install the packages:
    - to create virtual env: `python3 -m venv xplane-env --system-site-packages`
    - to activate virtual env: `source xplane-env/bin/activate`
    - to shut down virtual env: `deactivate`
    - make sure you add the name of your virtual environment to the .gitignore file! (xplane-env is already included)
4. build the aslxplane and xpc3 packages
    - ensure build package is up-to-date: `python3 -m pip install --upgrade build`
    - build packages: `python3 -m build`
5. install the aslxplane and xpc3 packages: `python3 -m pip install -e ./`
6. If you create new files or packages, uninstall, rebuild, and reinstall all the packages in the reopository:
    - uninstall packages `python3 -m pip uninstall ./`
    - rebuild packages (step 4)
    - reinstall packages (step 5)
