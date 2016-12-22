import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

case class X(id: String ,price: Double, lotsize: Double, bedrooms: Double,bathrms: Double,stories: Double, driveway: String,recroom: String,fullbase: String, gashw: String, airco: String, garagepl: Double, prefarea: String)


var input = "/home/manish/Desktop/Spark/data/Housing.csv"

val spark = SparkSession.builder.appName("LinearRegressionWithEncoding").master("local").getOrCreate()


val data = spark.sparkContext.textFile(input).map(_.split(",")).map( x => ( X(x(0), x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble,x(6), x(7), x(8), x(9), x(10), x(11).toDouble, x(12)))).toDF()

val categoricalVariables = Array("driveway","recroom", "fullbase", "gashw", "airco", "prefarea")

val categoricalIndexers: Array[org.apache.spark.ml.PipelineStage] =  categoricalVariables.map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index"))

val categoricalEncoders: Array[org.apache.spark.ml.PipelineStage] =  categoricalVariables.map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec"))

val assembler = new VectorAssembler().setInputCols( Array("lotsize", "bedrooms", "bathrms", "stories","garagepl","drivewayVec", "recroomVec", "fullbaseVec","gashwVec","aircoVec", "prefareaVec")).setOutputCol("features")


val lr = new LinearRegression().setLabelCol("price").setFeaturesCol("features").setMaxIter(1000).setSolver("l-bfgs")

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01, 0.001, 0.0001, 1.0)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 1.0)).build()

val steps = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)

val pipeline = new Pipeline().setStages(steps)

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator().setLabelCol("price")).setEstimatorParamMaps(paramGrid).setNumFolds(5)

val Array(training, test) = data.randomSplit(Array(0.75, 0.25), seed = 12345)

val model = cv.fit {
    training
}


val holdout = model.transform(test).select("prediction", "price").orderBy(abs(col("prediction")-col("price")))
holdout.show
val rm = new RegressionMetrics(holdout.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))
println(s"RMSE = ${rm.rootMeanSquaredError}")
println(s"R-squared = ${rm.r2}")

