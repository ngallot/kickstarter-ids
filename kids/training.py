import logging
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from mlflow.spark import log_model
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml import Pipeline, PipelineModel
from kids.utils import SparkUtils
from typing import List
from google.auth.compute_engine._metadata import _LOGGER as google_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kids.training')
google_logger.setLevel(logging.ERROR)


def build_string_indexer(col_name: str) -> StringIndexer:
    return StringIndexer()\
        .setInputCol(col_name)\
        .setOutputCol(f'{col_name}_indexed')\
        .setHandleInvalid("keep")


def build_one_hot_encoder(col_name: str) -> OneHotEncoder:
    return OneHotEncoder()\
        .setInputCol(col_name)\
        .setOutputCol(f'{col_name}_encoded')


def build_model(numerical_columns: List[str], categorical_columns: List[str], label_col: str, max_iter: int) -> Pipeline:

    indexing_stages = [build_string_indexer(c) for c in categorical_columns]
    indexed_columns = [s.getOutputCol() for s in indexing_stages]
    encoding_stages = [build_one_hot_encoder(c) for c in indexed_columns]

    vector_assembler = VectorAssembler() \
        .setInputCols(numerical_columns + [s.getOutputCol() for s in encoding_stages]) \
        .setOutputCol('features')

    gbt = GBTClassifier()\
        .setMaxIter(max_iter)\
        .setMaxDepth(6)\
        .setFeaturesCol(vector_assembler.getOutputCol())\
        .setLabelCol(label_col)

    return Pipeline()\
        .setStages(indexing_stages + encoding_stages + [vector_assembler, gbt])


def train(inputs_path: str):

    spark = SparkUtils.build_or_get_session('training')
    df_kids = spark.read.parquet(inputs_path)
    label_col = 'final_status'

    mlflow_tracking_ui = 'http://35.246.84.226'
    mlflow_experiment_name = 'kickstarter'

    mlflow.set_tracking_uri(mlflow_tracking_ui)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    numerical_columns = ['days_campaign', 'hours_prepa', 'goal']
    categorical_columns = ['country_clean', 'currency_clean']
    features = numerical_columns + categorical_columns

    df = df_kids.select(features + [label_col])

    max_iter = 15
    model_specs: Pipeline = build_model(numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                                        label_col=label_col, max_iter=max_iter)

    df_train, df_test = df.randomSplit([0.8, 0.2], seed=12345)
    df_train = df_train.cache()

    evaluator = BinaryClassificationEvaluator() \
        .setMetricName('areaUnderROC') \
        .setRawPredictionCol('rawPrediction') \
        .setLabelCol('final_status')

    with mlflow.start_run() as active_run:
        logger.info(f'Fitting model on {df_train.count()} lines')

        model: PipelineModel = model_specs.fit(df_train)

        logger.info('Evaluating model')
        train_metrics = evaluator.evaluate(model.transform(df_train))
        metrics = {'train_auc': train_metrics}

        test_metrics = evaluator.evaluate(model.transform(df_test))
        metrics.update({'test_auc': test_metrics})
        logger.info(f'Model metrics: {metrics}')

        logger.info('Logging to mlflow')
        mlflow_params = {'model_class': 'gbt', 'max_iter': max_iter}
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(metrics)
        log_model(model, 'model')
        model_uri = mlflow.get_artifact_uri(artifact_path='model')
        logger.info(f'Model successfully trained and saved @ {model_uri}')
