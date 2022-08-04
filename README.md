# Flockfysh: the data vending machine that gives more than it gets. 
flockfysh is an open source, efficient 2D-image tool that combines web scraping with artificial intelligence to generate and curate top quality image / object detection datasets. Feed flockfysh a "mini-dataset" with only ~50 images for each label (in the train category), and get back a hundredfold! 

We support your favorite tools such as [Roboflow](https://universe.roboflow.com/)!

We are currently looking for open source contributors, and would love to work with you to further develop this promising tool!

## How flockfysh works in a nutshell
We power up traditional object detection and classic imaging techniques such as data augmentations with data gathering techniques like lightning-fast web-scraping. The higher level algorithm (for most of our supported features is the same) functions is as described below:

General procedure for training and webscraping ("train-scrape")
1. Train object detection models on the small sample of data provided (images solely in the `train` folder will be considered)
2. Scrape various websites for images (based on a small number of searches queries supplied by the user), and use the model to figure out the most relevant and quality images 
3. Download those images, and train an *even better* object detection model 
4. Repeat steps 2-3 for some iterations, and then use the best model to gather the rest of the data

## How flockfysh reads and processes tasks
Flockfysh utilizes a seamless format to run its basic tasks. We support multiple workflows to generate *quality* datasets. Our core tool expects a small dataset with a [simple format](#flockysh-dataset-format) to begin with and an [input file](#flockfysh-input-format-inputyaml), and then we will automatically generate the rest of the dataset

## Quick start - what you need to run flockfysh
- A dataset (in the format specified below)
- A .yaml input file

### Flockfysh dataset format
All of the flockfysh operations support datasets in the format shown below: 

```
dataset/
    train/ (preferrably shift the most of your images here)
    valid/ (validation set, please keep a small number of images here)
    test/  (not needed by flockfysh)
```

We choose this format because it is the most consistent with the majority of Machine Learning and Computer Vision workflows, as well as *a structure that supports model training and testing right away*.

*Note that the dataset labels should be in yolo / yolov5 format*

### Flockfysh input format (`input.yaml`) 
In order to run, flockfysh requires a small amount of guidance from the user to help some information. More specifically, it needs:

1. The names of your classes for your dataset
2. The input directory name of your dataset folder (in the format [above](#flockfysh-dataset-format))
3. A Python dictionary mapping each class name to a list of search queries that would get you the images you want 

We can effectively provide this information to flockfysh by using an input YAML file. Additionally, there are customizable settings (such as training parameters and auxiliary options such as saving the bounding boxes for each image) that you can also toggle in this YAML format.

Flockfysh uses a general format in the input YAML as follows:

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

Each task ("train-scrape" is an example of a task or feature that flockfysh can perform) is treated as a seperate job that flockfysh can do. flockfysh supports multiple job operations, and performs each one in a sequential manner (i.e, does `job1`, then `job2`, etc). The identifier for each job (ex: `job1`) can be changed to whatever the user wants, and when running, flockfysh will notify the user via Terminal when it starts a job. For more information on the different kinds of jobs, [see the link below](#more-about-the-various-flockfysh-jobs)

For each job in YAML, there are a set of *mandatory* settings to include (take a look at the 3 settings above, for example). The job will **not** complete (and an error will be thrown) if those mandatory settings are not included. There are also other *default, configurable parameters settings* for each job that are adopted if they are not specified. Specifying them in the YAML for that job will override the defaults (for that job only).

To take a look at the default options / confing for a specific job, check out the [default settings folder].(https://github.com/teamnebulaco/flockfysh/tree/main/utilities/parse_config/default_params)


## A quick dive into running flockfysh
Make sure to check that you have everything specified in [link](#What-you-need). For most dataset generations, one can easily adapt a sample YAML workflow instead of needing to write one from scratch. 

Running flockfysh using the Github repo (latest code): 
1. Clone our repository by running the command `git clone https://github.com/teamnebulaco/flockfysh.git`
2. Run `cd flockfysh` to enter the repo and `pip install -r requirements.txt` to install the dependencies.
3. Export YoloV5 dataset (in format specified [above](#flockfysh-dataset-format)) into a folder inside the repository. If your dataset is on Roboflow, you have the option of exporting it and moving into the directory, or adding a download job at the beginning of the YAML to automatically load it in for you.
    - Don't have a dataset? Check out our [sample Roboflow dataset](https://github.com/teamnebulaco/sample-flockfysh-robo)
5. Create an `input.yaml` file (take a look at the sample `input.yaml` format)
6. Run `python run.py input.yaml` to run flockfysh with the specified input file `input.yaml`
    - On machines that use the command *python3* instead of *python* to execute Python 3, (change it to *python*)[https://stackoverflow.com/questions/23048756/how-can-i-make-the-python-command-in-terminal-run-python3-instead-of-python2] or use the command *python3 run.py input.yaml* 

## Sample Workflows

### Auto-Downloading a Roboflow Dataset and Running a Train-Scrape Job
For the purposes of this sample, we will use a [publicly available dataset](https://universe.roboflow.com/sanka-madushankaresearch/insectbite) within the Roboflow universe.

The sample YAML file for this workflow is the same as the example `input.yaml`. 

1. After cloning the repo and getting set up, add this into a `input.yaml` file at the base directory (same directory as `run.py`)

```
job1: #Download the dataset we want to train
  job-type: 'download'
  api-name: 'roboflow'
  api-key: 'ENTER_YOUR_API_KEY_HERE'
  workspace-name: "sanka-madushankaresearch"
  project-name: "insectbite"
  project-version: 1
  output-dirname: 'robo'
job2: #job1 can be replaced with any name for the job you prefer
  job-type: 'train-scrape'
  input-dir: robo
  class-names: ['Bed Bug', 'Fire ant', 'Tick', 'Wasp']
  class-search-queries: {'Bed Bug' : ['bed bug bites'], 'Fire ant' : ['fire ant bites'], 'Tick' : ['tick bites'], 'Wasp' : ['wasp bites']}
  train-workers: 8
  images-per-label: 500
  total-maximum-images: 7000
  image-dimensions: 200
  train-batch: 8

```

2. Replace `api-key` with the API key you get from Roboflow (can easily be found when exporting a dataset using code).
3. Run `python run.py input.yaml` to run the workflow!

### Using a custom dataset to run a Train-Scrape Job
For the purposes of this sample, we will use a [publicly available dataset](https://universe.roboflow.com/sanka-madushankaresearch/insectbite) within the Roboflow universe, but locally download it. The dataset should be in the format specified [above](#flockfysh-dataset-format).


1. After cloning flockfysh and getting set up, run `git clone https://github.com/teamnebulaco/sample-flockfysh-robo.git` and move the `robo` folder inside into the base directory
2. Add the code below into a `input.yaml` file at the base directory (same directory as `run.py`)

```
job1: #job1 can be replaced with any name for the job you prefer
  job-type: 'train-scrape'
  input-dir: robo
  class-names: ['Bed Bug', 'Fire ant', 'Tick', 'Wasp']
  class-search-queries: {'Bed Bug' : ['bed bug bites'], 'Fire ant' : ['fire ant bites'], 'Tick' : ['tick bites'], 'Wasp' : ['wasp bites']}
  train-workers: 8
  images-per-label: 500
  total-maximum-images: 7000
  image-dimensions: 200
  train-batch: 8

```

3. Run `python run.py input.yaml` to run the workflow!

## More about the various flockfysh jobs 
You must specify which type of job you want for each using a `job-type` attribute. Here are the different job types available.

### download
Automatically downloads a dataset from a specific API. The current APIs supported are listed below:
- Roboflow

Note that each API also has API-specific information (API keys, secrets, etc) that flockfysh needs to utilize to download the dataset.

### train-scrape
Utilizes object detection models in tandem with web-scraping to generate an image-dataset. Here are some relevant properties:

- `input-dir`: The path to the dataset folder 
- `class-names`: An array of labels for the images you are trying to classify 
- `class-search-queries`: An array of words you'd put in a search (imagine Googling the images yourself) to download the images.
- `train-workers`: Number of workers for training
- `images-per-label`: MAIN PROPERTY TO control dataset size - how many images you want for each of the class names specified in `class-names`
- `total-maximum-images`: Adds an upper limit on the number of images (most support is for `images-per-label` at the moment)

## Development / Open Source
We are extremely excited to open this repository to the community, and can't wait to see the future to which this project heads! Please consider joining our Discord server, which we use as our main platform to communicate, improve, and resolve issues.

Steps to begin development: 
1. Clone our repository by running the command `git clone https://github.com/teamnebulaco/flockfysh.git`
2. Switch into our current dev_branch by running `git checkout -b dev-branch`
3. Pull the code. `git pull origin dev-branch`
3. Create a virtualenv by running the command `python -m virtualenv venv` (Syntax may vary)
    1. Note that the code above assumes **you have the virtualenv package installed** If not, run the command `pip install --upgrade virtualenv`
4. Run `python run.py input.yaml` to start developing! Happy coding!

## License
This repository is licensed under the [BSD-4 license](LICENSE.md). **Note that some of the images from the scraper may have artisitic copyrights, and should only, only, ONLY be used for ML & Training purposes. Under no grounds should this tool be exploited to circumvent copyrights.** Besides, it makes everyone's lives easier if we don't mooch off each other's copyrighted images. :))
