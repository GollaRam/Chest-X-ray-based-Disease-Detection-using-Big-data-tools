import org.apache.spark.sql.{SparkSession, DataFrame}

object CleanCSV {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("CSV Cleaning Example")
      .master("local[*]") 
      .getOrCreate()
    val inputPath = "hdfs://localhost:9000/chestxray/train.csv"
    val outputPath = "hdfs://localhost:9000/chestxray/train_cleaned_single.csv"
    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)
    val labelColumns = Seq(
      "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
      "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
      "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    )
    df = df.na.fill(0.0, labelColumns)
    df = df.na.fill("LA", Seq("AP/PA"))
    df.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(outputPath)

    println(s"leaned data written to $outputPath as a single CSV file")

    spark.stop()
  }
}
