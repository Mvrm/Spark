import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions._


import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder.appName("KMeansExample").master("local").getOrCreate()

val ds = spark.read.option("inferSchema", "true").option("header", "true").option("nullValue", "?").csv("/home/manish/Desktop/Spark/data/mtcars.csv")

ds.printSchema()
ds.show()

val assembler = new VectorAssembler().setInputCols(Array("mpg", "cyl", "disp", "hp", "drat", "wt")).setOutputCol("features")
     
val assemdata = assembler.transform(ds)
     
val scaled = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)

val scalerModel = scaled.fit(assemdata)
 
val scaledData = scalerModel.transform(assemdata)
  
val clusters = 10

val kmeans = new KMeans().setK(clusters).setMaxIter(1000).setFeaturesCol("scaledFeatures").setPredictionCol("prediction")
       
val model = kmeans.fit(scaledData)
  

val WSSSE = model.computeCost(scaledData)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)

val predict = model.transform(scaledData)
predict.show(1000)
    
for (i <- 0 to clusters) { 
        val predictionsPerCol = predict.filter(col("prediction") === i)
        println(s"Cluster $i")
       predictionsPerCol.select(col("_c0"), col("features"), col("prediction")).collect.foreach(println)
       println("======================================================")
    }

    spark.stop()


