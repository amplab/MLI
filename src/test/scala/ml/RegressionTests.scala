package mli.test.ml


import mli.interface.MLContext
import mli.test.testsupport.LocalSparkContext
import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter
import org.jblas.DoubleMatrix
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.util.Random

class RegressionTests extends FunSuite with BeforeAndAfter with LocalSparkContext {

  def make_synthetic_data(nexamples: Int, nfeatures: Int, eps: Double = 0.01) = {
    val w = DoubleMatrix.rand(nfeatures, 1)
    val X = DoubleMatrix.rand(nexamples, nfeatures)
    val y = X.mmul(w)
    val rnd = (1 to nexamples).map(_ => Random.nextGaussian())
    val yObs = y.add(new DoubleMatrix(rnd.toArray))
    val tbl = (0 until nexamples).map( i=> (yObs.get(i, 0), X.getRow(i).data))
    val tblRDD = sc.parallelize(tbl, 64)
    (w, tblRDD)
  }


  test("Testing RidgeRegression") {
    sc = new SparkContext("local", "test")
    val mc = new MLContext(sc)

    /*
    val (wTrue, tblRDD) = make_synthetic_data(10000, 100, 0.1)
    val model = RidgeRegression.train(tblRDD)
    println("Created synthetic dataset")
    println("\t Examples: " + tblRDD.count())
    println("\t Features: " + tblRDD.take(1)(0)._2.length)    
    println("\t True Weight: ")
    println(wTrue)
    println("\t Learned model: ")
    println(model.wOpt)    
    */
  }


}
