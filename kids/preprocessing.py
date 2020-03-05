from typing import Optional
from pyspark.sql import SparkSession, DataFrame, functions as fn, types as st


class Udfs:

    @staticmethod
    def country(country: str, currency: str) -> str:
        return currency if country is False else country

    @staticmethod
    def currency(currency: str) -> Optional[str]:
        return currency if len(currency) == 3 else None

    @staticmethod
    def is_valid_label(label: int) -> bool:
        return label in [0, 1]

    @staticmethod
    def filter_hours_days_goal(hours_prepa, days_campaign, goal) -> bool:
        def _gt(value) -> bool:
            try:
                return value >= 0
            except:
                return False
        return _gt(hours_prepa) and _gt(days_campaign) and _gt(goal)


def build_training_set(inputs: DataFrame) -> DataFrame:

    udf_country = fn.udf(Udfs.country, st.StringType())
    udf_currency = fn.udf(Udfs.currency, st.StringType())
    udf_is_valid_label = fn.udf(Udfs.is_valid_label, st.BooleanType())

    udf_filter = fn.udf(Udfs.filter_hours_days_goal, st.BooleanType())

    replace_values = {'days_campaign': -1,
                      'hours_prepa': -1,
                      'goal': -1,
                      'country_clean': 'unknown',
                      'currency_clean': 'unknown'
                      }

    result = inputs.withColumn('goal', fn.col('goal').cast(st.DoubleType())) \
        .withColumn('deadline', fn.col('deadline').cast(st.IntegerType())) \
        .withColumn('state_changed_at', fn.col('state_changed_at').cast(st.IntegerType())) \
        .withColumn('created_at', fn.col('created_at').cast(st.IntegerType())) \
        .withColumn('launched_at', fn.col('launched_at').cast(st.IntegerType())) \
        .drop('disable_communication') \
        .drop('state_changed_at', 'backers_count') \
        .withColumn('country_clean', udf_country(fn.col('country'), fn.col('currency'))) \
        .withColumn('currency_clean', udf_currency(fn.col('currency'))) \
        .drop('country', 'currency') \
        .filter(udf_is_valid_label(fn.col('final_status'))) \
        .withColumn("deadline_clean", fn.to_date(fn.from_unixtime(fn.col('deadline')))) \
        .withColumn("created_at_clean", fn.to_date(fn.from_unixtime(fn.col('created_at')))) \
        .withColumn("launched_at_clean", fn.to_date(fn.from_unixtime(fn.col('launched_at')))) \
        .withColumn("days_campaign", fn.datediff(fn.col('deadline_clean'), fn.col('launched_at_clean'))) \
        .withColumn("hours_prepa", fn.round((fn.col('launched_at') - fn.col('created_at')) / 3600, 2)) \
        .filter(udf_filter(fn.col('hours_prepa'), fn.col('days_campaign'), fn.col('goal'))) \
        .drop('created_at', 'launched_at', 'deadline') \
        .withColumn("name", fn.lower(fn.col('name'))) \
        .withColumn("desc", fn.lower(fn.col('desc'))) \
        .withColumn("keywords", fn.lower(fn.col('keywords'))) \
        .withColumn("text", fn.concat_ws(" ", fn.col('name'), fn.col('desc'), fn.col('keywords'))) \
        .drop("name", "desc", "keywords") \
        .na.fill(replace_values)

    return result
