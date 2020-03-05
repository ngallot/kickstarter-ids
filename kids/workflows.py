from kids.utils import SparkUtils
from kids.preprocessing import build_training_set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kids.workflows')


def preprocess(inputs_path: str, destination_path: str):
    try:
        spark = SparkUtils.build_or_get_session(app_name='preprocessing')
        inputs = spark.read.csv(inputs_path, inferSchema=True, header=True)
        preprocessed = build_training_set(inputs=inputs)
        preprocessed \
            .repartition(8*10) \
            .write \
            .mode('overwrite') \
            .parquet(destination_path)
        print(f'File exported @ {destination_path}')
    except Exception as e:
        logger.error(f'Error while preprocessing dataset: {e}')