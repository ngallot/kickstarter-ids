# kickstarter-ids
An ML based package to predict the success of a Kickstarter campaign.

### Quickstart

#### Installation
Create a virtual environment and install the dependencies in it:
```bash
virtualenv venv
source venv/bin/activate
pip install .
```

#### Dataset preprocessing

Check the help of the included command line:
```bash
kids --help
```
```bash
Usage: kids [OPTIONS] COMMAND [ARGS]...

  A command line tool to train a kickstarter campaign success predictor

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  preprocess  Applies preprocessing steps to the input data and build...
```
Transform the input data into a training dataset in parquet format:
```bash
kids preprocess -i path_to_you_file.csv -d path_to_destination.parquet 
```
