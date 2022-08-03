# Flockfysh
The vending machine that gives more than it gets.

## How to run Flockfysh
NOTE that the instructions below are subject to change, this is a rapidly developing tool

Steps to Run: 
1. Clone our repository by running the command `git clone https://github.com/teamnebulaco/flockfysh.git`
2. Export Roboflow dataset into Yolov5 format into a folder inside the repository. Don't have a dataset? Check out our [sample dataset](https://github.com/teamnebulaco/sample-flockfysh-robo)
3. Create an `input.yaml` file (take a look at the sample `input.yaml` format)
3. Run `python3 run.py input.yaml` to run

## Development
Steps to begin development: 
1. Clone our repository by running the command `git clone https://github.com/teamnebulaco/flockfysh.git`
2. Switch into our current dev_branch by running `git checkout -b dev-branch`
3. Pull the code. `git pull origin dev-branch`
3. Create a virtualenv by running the command `virtualenv -p python3 envname`
    1. Note that the code above assumes **you have the virtualenv package installed** If not, run the command `pip install --upgrade virtualenv`
4. Run `python3 run.py input.yaml` to start developing! Happy coding!

