import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Training").getOrCreate()

csv_schema = StructType([
    StructField("fixed_acidity", DoubleType()),
    StructField("volatile_acidity", DoubleType()),
    StructField("citric_acid", DoubleType()),
    StructField("residual_sugar", DoubleType()),
    StructField("chlorides", DoubleType()),
    StructField("free_sulfur_dioxide", DoubleType()),
    StructField("total_sulfur_dioxide", DoubleType()),
    StructField("density", DoubleType()),
    StructField("pH", DoubleType()),
    StructField("sulphates", DoubleType()),
    StructField("alcohol", DoubleType()),
    StructField("quality", DoubleType())
])

wine_dataset = spark.read.format("csv").schema(csv_schema).options(header=True, delimiter=';', quote='"', ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True).load('file:///home/ec2-user/TrainingDataset.csv')
wine_dataset = wine_dataset.toDF(*[column.replace('"', '') for column in wine_dataset.columns])
wine_dataset = wine_dataset.withColumn("quality", F.when(F.col("quality") > 7, 1).otherwise(0))

feature_columns = wine_dataset.columns[:-1]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
wine_dataset = vector_assembler.transform(wine_dataset)

training_data, testing_data = wine_dataset.randomSplit([0.8, 0.2])

svm_classifier = LinearSVC(labelCol="quality", featuresCol="features")
trained_svm_model = svm_classifier.fit(training_data)

test_predictions = trained_svm_model.transform(testing_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(test_predictions)
print("Evaluated F1 Score: {:.4f}".format(f1_score))

trained_svm_model.save("file:///home/ec2-user/LinearSVC")
