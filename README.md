# flockfysh
The vending machine that gives more than it gets. Give it a little whiff of the data you want by providing a "mini-dataset" with only ~50 images for each label (in the train category), and get back tenfold! For object detection, we support your favorite tools such as [Roboflow](https://universe.roboflow.com/)

## How flockfysh works in a nutshell
We power up traditional object detection and classic imaging techniques such as data augmentations with data gathering techniques like lightning-fast web-scraping. The higher level algorithm (for most of our supported features is the same) functions as described below:

General Procedure for Training and Webscraping ("train-scrape")
1. Train object detection models on the small sample of data provided (images solely in the `train` folder will be considered)
2. Scrape various websites for images (based on a small number of searches queries supplied by the user), and use the model to figure out the most relevant and quality images 
3. Download those images, and train an *even better* object detection model 
4. Repeat steps 2-3 for some iterations, and then use the best model to gather the rest of the data

## How flockfysh reads and processes tasks
flockfysh utilizes a seamless format to run it's basic tasks. We support multiple workflows, and make it simple to get *quality* datasets. Our core tool expects a small dataset with a simple format:

## Quick Start - What you need to run flockfysh
- A dataset (in the format specified below)
- A .yaml input file

### flockfysh Dataset Format
All of the flockfysh operations support datasets in the format shown below: 

```
dataset/
    train/ (preferrably shift the most of your images here)
    valid/ (validation set, please keep a small number of images here)
    test/  (not needed by flockfysh)
```

We choose this format because we find that this dataset format is the most consistent with the majority of Machine Learning (ML) & Computer Vision (CV) workflows, as well as *a structure that supports model training and testing right away*.

### flockfysh Input Format (`input.yaml`) 
In order to run, flockfysh requires a small amount of guidance from the user to help some information. More specifically, it needs:

1. The names of your classes for your dataset
2. The input directory name of your dataset folder (in the format above)
3. A Python dictionary mapping each class name to a list of search queries that would get you the images you want 

We can effectively provide this information to flockfysh by using an input YAML file. Additionally, there are customizable settings (such as training parameters and auxiliary options such as saving the bounding boxes for each image) that you can also toggle in this YAML format.

flockfysh uses a general format in the input YAML as follows:

```
job1:
    job-type: 'train-scrape'
    input-dir:
    ...
job2:
    ...
jobn:
    ...
```

Each task ("train-scrape" is an example of a task or feature that flockfysh can perform) is treated as a seperate job that flockfysh can do. flockfysh supports multiple job operations, and performs each one in a sequential manner (i.e, does `job1`, then `job2`, etc). The identifier for each job (ex: `job1`) can be changed to whatever the user wants, and when running, flockfysh will notify the user via Terminal when it starts a job.

For each job in YAML, there are a set of *mandatory* settings to include (take a look at the 3 settings above, for example). The job will **not** be able to be completed (and an error will be thrown) if those mandatory settings aren't included. There are also other *default, configurable parameters settings* for each job that are adopted if they aren't specified. Specifying them in the YAML for that job will override the defaults (for that job only).

If you'd like to take a look at the default options / confing for a specific job, check out the [default settings folder].(https://github.com/teamnebulaco/flockfysh/tree/main/utilities/parse_config/default_params)

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

