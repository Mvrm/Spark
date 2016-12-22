import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types._

import org.apache.spark.sql.SparkSession


val spark = SparkSession.builder.appName("SpamExample").master("local").getOrCreate()

    val customSchema = StructType(Array(StructField("spam", StringType, true),StructField("message", StringType, true)))

    val ds = spark.read.option("inferSchema", "true").option("delimiter", "\t").schema(customSchema).csv("/home/manish/Desktop/Spark/data/SMSSpamCollection.tsv")

    ds.printSchema()

    ds.show(8)

    val indexer = new StringIndexer().setInputCol("spam").setOutputCol("label")
    val indexed = indexer.fit(ds).transform(ds)

    indexed.show()

    
    val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("tokens")
    val tokdata = tokenizer.transform(indexed)

    tokdata.show()

    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("tf")
    val tfdata = hashingTF.transform(tokdata)

    tfdata.show()

    val idf = new IDF().setInputCol("tf").setOutputCol("idf")
    val idfModel = idf.fit(tfdata)
    val idfdata = idfModel.transform(tfdata)

    val assembler = new VectorAssembler().setInputCols(Array("idf")).setOutputCol("features")
    val assemdata = assembler.transform(idfdata)

    val Array(trainingData, testData) = assemdata.randomSplit(Array(0.7, 0.3), 1000)

    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")


    val lrModel = lr.fit(trainingData)

    val str = lrModel.toString()
    println(str)

    val predict = lrModel.transform(testData)
    predict.show(100)

    val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(predict)
    println("Test Error = " + (1.0 - accuracy))

    spark.stop()