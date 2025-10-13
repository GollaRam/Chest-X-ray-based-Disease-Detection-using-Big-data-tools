import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.java.{XGBoost => JXGBoost, DMatrix, Booster}
import scala.collection.JavaConverters._
import java.io.PrintWriter
import java.util.Base64

object XGBoost {

  val labelFields = Array(
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
  )

  def main(args: Array[String]): Unit = {
    val trainParquetPath = "hdfs://localhost:9000/chestxray/embeddings_final1.parquet"
    val modelOutputDir = "hdfs://localhost:9000/chestxray/xgb_model_trail1"

    val spark = SparkSession.builder()
      .appName("XGBoost Multilabel Medical Classification")
      .master("local[*]")
      .config("spark.driver.memory", "8g")
      .config("spark.executor.memory", "8g")
      .config("spark.driver.maxResultSize", "4g")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("Loading data from Parquet...")
    val df = loadParquet(spark, trainParquetPath, labelFields)
    val assembledDf = assembleFeatures(df, 768)

    val Array(trainDf, validDf) = assembledDf.randomSplit(Array(0.8, 0.2), 42)
    println("Data loaded and split successfully!")

    val thresholds = labelFields.map(label => trainLabel(label, trainDf, validDf, modelOutputDir, spark))
    

    println("XGBoost training complete!")
    spark.stop()
  }

  def loadParquet(spark: SparkSession, path: String, labels: Array[String]): DataFrame = {
    val df = spark.read.parquet(path)
    labels.foldLeft(df)((d, l) => d.withColumn(l, col(l).cast(DoubleType)))
      .withColumn("Sex", col("Sex").cast(DoubleType))
      .withColumn("Age", col("Age").cast(DoubleType))
  }

  def assembleFeatures(df: DataFrame, embSize: Int): DataFrame = {
    val featureCols = (0 until embSize).map(i => s"emb_$i").toArray ++ Array("Sex", "Age")
    new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")
      .transform(df)
  }

  def trainLabel(label: String, trainDf: DataFrame, validDf: DataFrame, modelDir: String, spark: SparkSession): Double = {
    println(s"Training model for label: $label")

    val (trainMatrix, trainLabels) = toDMatrix(trainDf, label)
    val (validMatrix, validLabels) = toDMatrix(validDf, label)

    val posCount = trainLabels.count(_ == 1f)
    val negCount = trainLabels.count(_ == 0f)
    val scalePos = if (posCount > 0) negCount.toFloat / posCount.toFloat else 1.0f

    val params = Map(
      "objective" -> "binary:logistic",
      "eval_metric" -> "logloss",
      "scale_pos_weight" -> scalePos.toString,
      "max_depth" -> "6",
      "eta" -> "0.05",
      "subsample" -> "0.8",
      "colsample_bytree" -> "0.8",
      "verbosity" -> "0",
      "seed" -> "42",
      "nthread" -> "4"
    ).map { case (k, v) => k -> (v: Object) }.asJava

    val booster = JXGBoost.train(trainMatrix, params, 500, Map("train" -> trainMatrix).asJava, null, null)

    val probsAndLabels = booster.predict(validMatrix).map(_.head.toDouble).zip(validLabels.map(_.toDouble))
    val bestThreshold = findBestThreshold(probsAndLabels)
    val baos = new java.io.ByteArrayOutputStream()
    booster.saveModel(baos)
    val modelBase64 = Base64.getEncoder.encodeToString(baos.toByteArray)
    val labelPath = s"$modelDir/xgb_model_${label.replace(" ", "_")}.model"
    spark.sparkContext.parallelize(Seq(modelBase64), 1).saveAsTextFile(labelPath)
    println(s"Model for '$label' saved successfully at $labelPath")
    bestThreshold
  }

  def toDMatrix(df: DataFrame, label: String): (DMatrix, Array[Float]) = {
    val data = df.select("features", label).collect()
    val features = data.flatMap { r =>
      r.getAs[DenseVector]("features").toArray.map(_.toFloat)
    }
    val labels = data.map(_.getDouble(1).toFloat)
    val numRows = data.length
    val numCols = features.length / numRows
    val matrix = new DMatrix(features, numRows, numCols)
    matrix.setLabel(labels)
    (matrix, labels)
  }

  def findBestThreshold(probsAndLabels: Array[(Double, Double)]): Double = {
    val candidates = (1 to 200).map(i => 0.001 + (i - 1) * (0.99 - 0.001) / 199.0)
    candidates.maxBy { t =>
      val (tp, fp, tn, fn) = calculateConfusionMatrix(probsAndLabels, t)
      calculateF1(tp, fp, fn)
    }
  }

  def calculateConfusionMatrix(probsAndLabels: Array[(Double, Double)], threshold: Double): (Int, Int, Int, Int) = {
    var tp, fp, tn, fn = 0
    probsAndLabels.foreach { case (prob, label) =>
      val pred = if (prob >= threshold) 1.0 else 0.0
      if (pred == 1.0 && label == 1.0) tp += 1
      else if (pred == 1.0 && label == 0.0) fp += 1
      else if (pred == 0.0 && label == 0.0) tn += 1
      else if (pred == 0.0 && label == 1.0) fn += 1
    }
    (tp, fp, tn, fn)
  }

  def calculateF1(tp: Int, fp: Int, fn: Int): Double = {
    val precision = if (tp + fp > 0) tp.toDouble / (tp + fp) else 0.0
    val recall = if (tp + fn > 0) tp.toDouble / (tp + fn) else 0.0
    if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0
  }


}
