import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.SparkConf


object Cluster {
  def main(args: Array[String]): Unit = {
     val conf = new SparkConf().setAppName("Cluster").setMaster("local")
     val sc = new SparkContext(conf)
     val data = sc.textFile("hdfs://happygong:9000/cloud_competition/data/kdd_modify*")
     val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
     //取消量纲
     val normData = new Normalizer().transform(parsedData).cache()
     //normData.foreach { a => println(a.toString()) }
     //将数据划分成23个簇
     val numClusters = 23
     val numIterations = 100    //迭代100次
     val clusters = KMeans.train(normData, numClusters, numIterations)
     val WSSSE = clusters.computeCost(normData)
     println("Within Set Sum of Squared Errors=" + WSSSE)
     val ss = parsedData.map(v => clusters.predict(v)).collect()
     ss.foreach { a => println(a.toString()) }
  }
}
