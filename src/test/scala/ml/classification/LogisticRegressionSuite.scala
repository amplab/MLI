package mli.test.ml.classification

import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter
import mli.interface._
import mli.ml.classification.LogisticRegressionAlgorithm
import mli.test.testsupport.LocalSparkContext
import org.apache.spark.SparkContext

class LogisticRegressionSuite extends FunSuite with LocalSparkContext {
  val sampleData = Array(Array(1.0,10.0,12.0), Array(1.0,12.0,15.0), Array(-1.0,1.0,2.0), Array(-1.0,2.0,1.0))

  test("Basic test of logistic regression") {
    sc = new SparkContext("local", "test")
    val mc = new MLContext(sc)


    val data = mc.load(sampleData)
    val model = LogisticRegressionAlgorithm.train(data)

    println(model.explain)

    var x = model.predict(MLVector(Array(20.0,20.0))).toNumber
    println("Model prediction for (20.0,20.0): " + x)
    assert(x > 0.5)

    x = model.predict(MLVector(Array(0.0,0.0))).toNumber
    println("Model prediction for (-1.0,0.0): " + x)
    assert(x <= 0.5)
  }

  test("Basic test of logistic regression via Parallel Gradient") {
    sc = new SparkContext("local", "test")
    val mc = new MLContext(sc)

    val data = mc.load(sampleData)
    val params = LogisticRegressionAlgorithm.defaultParameters().copy(optimizer = "Gradient")
    val model = LogisticRegressionAlgorithm.train(data, params)

    println(model.explain)
    assert(true)
  }

}
