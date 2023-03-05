The ASL's port of the NASA-ULI-Xplane repository. Refactored the ULI-Xplane repo to improve code quality, readability, and functionality

## To Do:
- [ ] create one file with code to run one trajectory, including data logging
- [ ] create one file template to run many experiments
- [ ] move all config files to json/yaml
- [ ] create controller abstract template
- [ ] optionally, create ros pakage to run experiments with the simulator with ros
- [ ] 

## Setup Instructions:
1. Clone the repository: `git clone https://github.com/StanfordASL/XPlane-ASL.git`
2. Enter the XPLANE-ASL directory: `cd XPLANE-ASL`
3. (optional) create and activate a virtual environment to install the packages:
  - to create virtual env: 'python3 -m venv xplane-asl --system-site-packages'
  - to activate virtual env: `source xplane-env/bin/activate`
  - to shut down virtual env: `deactivate`
4. build the aslxplane and xpc3 packages
  - ensure build package is up-to-date: `python3 -m pip install --upgrade build`
  - build packages: `python3 -m build`
5. install the aslxplane and xpc3 packages: `python3 -m pip install -e ./`
6. If you create new files or packages, uninstall, rebuild, and reinstall all the packages in the reopository:
  - uninstall packages `python3 -m pip uninstall ./`
  - rebuild packages (step 4)
  - reinstall packages (step 5)
