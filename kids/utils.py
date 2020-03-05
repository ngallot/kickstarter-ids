class SparkUtils:

    @staticmethod
    def build_or_get_session(app_name: str):
        from pyspark.sql import SparkSession
        return SparkSession.builder.appName(app_name).getOrCreate()
