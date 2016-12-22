import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types._

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder.appName("ChurnExample").master("local").getOrCreate()
	  
val customSchema = StructType(Array(
      StructField("state", StringType, true),
      StructField("account_length", DoubleType, true),
      StructField("area_code", StringType, true),
      StructField("phone_number", StringType, true),
      StructField("intl_plan", StringType, true),
      StructField("voice_mail_plan", StringType, true),
      StructField("number_vmail_messages", DoubleType, true),
      StructField("total_day_minutes", DoubleType, true),
      StructField("total_day_calls", DoubleType, true),
      StructField("total_day_charge", DoubleType, true),
      StructField("total_eve_minutes", DoubleType, true),
      StructField("total_eve_calls", DoubleType, true),
      StructField("total_eve_charge", DoubleType, true),
      StructField("total_night_minutes", DoubleType, true),
      StructField("total_night_calls", DoubleType, true),
      StructField("total_night_charge", DoubleType, true),
      StructField("total_intl_minutes", DoubleType, true),
      StructField("total_intl_calls", DoubleType, true),
      StructField("total_intl_charge", DoubleType, true),
      StructField("number_customer_service_calls", DoubleType, true),
      StructField("churned", StringType, true)))

val ds = spark.read.option("inferSchema", "true").schema(customSchema).csv("~/data/churn.all")
ds.printSchema()

val indexer = new StringIndexer().setInputCol("intl_plan").setOutputCol("intl_plan_idx")
val indexed = indexer.fit(ds).transform(ds)

indexed.printSchema()

val churn = new StringIndexer().setInputCol("churned").setOutputCol("churned_idx")
val churned = churn.fit(indexed).transform(indexed)

val assembler = new VectorAssembler().setInputCols(Array("account_length", "intl_plan_idx", "number_vmail_messages", "total_day_minutes",
                          "total_day_calls", "total_day_charge", "total_eve_minutes", "total_eve_calls",
                          "total_night_minutes", "total_night_calls", "total_night_charge", "total_intl_minutes",
                          "total_intl_calls", "total_intl_charge", "number_customer_service_calls")).setOutputCol("features")

val 	 = assembler.transform(churned)
assemdata.printSchema()

val Array(trainingData, testData) = assemdata.randomSplit(Array(0.7, 0.3), 1000)

val rf = new RandomForestClassifier().setLabelCol("churned_idx").setFeaturesCol("features").setNumTrees(10)

val rfModel = rf.fit(trainingData)
val str = rfModel.toDebugString
println(str)

val predict = rfModel.transform(testData)
predict.select("churned", "prediction").show(1000)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("churned_idx").setRawPredictionCol("prediction")

val accuracy = evaluator.evaluate(predict)
println("Test Error = " + (1.0 - accuracy))

spark.stop()