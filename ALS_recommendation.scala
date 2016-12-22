import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

import org.apache.spark.sql.SparkSession

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}


val spark = SparkSession.builder.appName("ALSExample").master("local").getOrCreate()

import spark.implicits._

val ratings = spark.read.textFile("/home/manish/Desktop/Spark/data/movielens_ratings.txt").map(parseRating).toDF()

val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)

val predictions = model.transform(test)

val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

spark.stop()