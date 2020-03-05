import click
from kids import __version__


@click.group(name='kids', help='A command line tool to train a kickstarter campaign success predictor')
@click.version_option(__version__)
def kids():
    pass


@kids.command(name='preprocess', help='Applies preprocessing steps to the input data and build training set')
@click.option('--inputs-path', '-i', help='The path inputs can be found')
@click.option('--destination-path', '-d', help='The path where to write the preprocessed file')
def preprocess(inputs_path: str, destination_path: str):
    from kids import workflows as wks
    wks.preprocess(inputs_path=inputs_path, destination_path=destination_path)


@kids.command(name='train', help='Train a classifier model to predict success or failure of a Kickstarter campaign')
@click.option('--parquet-path', '-p', help='The where input parquet file can be found')
def train(parquet_path: str):
    from kids import workflows as wks
    wks.train(parquet_file_path=parquet_path)
