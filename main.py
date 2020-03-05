from pyspark.sql import SparkSession
from kids import preprocessing

if __name__ == '__main__':
    spark: SparkSession = SparkSession.builder.getOrCreate()

    inputs_path = '/home/ngallot/git/teaching/vivadata/vivadata-inphb-challenges-mywork/04-kickstarter-preprocessing/data/train_clean.csv'
    inputs = spark.read.csv(inputs_path, inferSchema=True, header=True)
    preprocessed = preprocessing.build_training_set(inputs=inputs)

    print(f'nb lines: {preprocessed.count()}')


