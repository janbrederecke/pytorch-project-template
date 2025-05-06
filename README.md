
---

# PyTorch Project Template

This repository offers a structured framework for PyTorch projects. It was highly influenced by [Christof Henkel](https://github.com/ChristofHenkel)'s public 
repositories.

The template contains example model, dataset etc. for the TLVMC-Freezing-of-Gait-Competition and can be run with the competition-data 
in place.

It is best used in combination with my automated [Kaggle container setup](https://github.com/janbrederecke/kaggle_training_container) 
and the included local MLflow instance.

Automated logging of models to the MLflow model registry is still a work in progress. By now, only state_dicts are saved as artifacts to MLflow.
Some other details are also still being worked on to create the best training-experience.

## Necessary Prerequisites

- You know how to use a [Makefile](https://opensource.com/article/18/8/what-how-makefile)
- [Screen](https://www.gnu.org/software/screen/manual/screen.html) to keep your training running even when you disconnect from your SSH session.

## Repository Structure

| **File/Folder Name**   | **Description**                                                              |
|------------------------|------------------------------------------------------------------------------|
| `configs`               | Configuration files for the training reside here                               |
| `src`                  | Contains all the python code from training, model, dataset to post-processing|
| `main.py`              | Runs the actual training program                                             |
| `Makefile`              | Contains all commands to call the main.py file in different ways needed       |
| `preprocessing.ipynb`  | Jupyter notebook for preprocessing data to the right format needed here      |

## Configuration / How to get started

Individual files added to *configs* can be used for training a model. All you have to do is change the paths and paramenters 
and add the file name to the top of the Makefile or your call to the Makefile.

The given config file will then be used when a training run is started.

Generally speaking, all you have to do is preprocess your data so that it can be loaded by a dataset that you provide, and pass it 
to a suitable model (take a look in into the examples to understand the specifics in this setup). Besides that, you can now automate
the whole training process with the config.py file used.

Data should generally reside in a folder called data. When using my automated container setup, the raw competition data is
downloaded and mounted under data automatically.

To start a training run, use the commands from the Makefile explained below.

## Using the Makefile

1. **Run training on all folds consecutively**  
   Command:  
   ```bash
   make all_folds CONFIG=<your_config>
   ```  
   This command trains the model on all data folds one by one, based on the specified configuration.

2. **Run training on the first fold only**  
   Command:  
   ```bash
   make one_fold CONFIG=<your_config>
   ```  
   This command trains the model on the first fold using the specified configuration.

3. **Run training on all folds combined, with validation on fold 0**  
   Command:  
   ```bash
   make all_data CONFIG=<your_config>
   ```  
   This command trains the model on the full dataset and validates using fold 0.

### Notes
- Replace `<your_config>` with the name of the configuration file (without the `.py` extension) located in the `configs` directory. For example, if your configuration file is `configs/config.py`, use `CONFIG=config`.
- The Makefile dynamically determines whether to run distributed training or single-GPU training based on the `DISTRIBUTED` option in your configuration file.

This manual ensures a simplified approach to running various training scenarios with minimal setup effort.

