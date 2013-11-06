package mli.util

import mli.interface._
import mli.interface.impl.MLNumericTable
import mli.ml.classification._
import org.apache.spark.SparkContext

object MLILRDemo {
  def main(args: Array[String]) = {
    val sc = new SparkContext("local[4]", "MLIRtest")
    val mc = new MLContext(sc)

    val data = mc.loadCsvFile(args(0))
    val d2 = data.map((x: MLRow) => x.drop(0).+:(if(x(0).toString == "n07760859") MLValue(1.0) else MLValue(0.0))).cache()
    val d3 = MLNumericTable(d2.toDoubleArrayRDD()).cache()
    val model = LogisticRegressionAlgorithm.train(d3, LogisticRegressionParameters())
    //val model = SVMAlgorithm.train(d2)

    println("Time to train: " + model.trainingTime)
    sc.stop()
  }
}