//package mli.test.interface
//
//import org.scalatest.FunSuite
//import scala.util.Random
//import mli.test.testsupport.LocalSparkContext
//
//class MLTableSuite extends FunSuite with LocalSparkContext {
//  def makeToyLRData() = {
//
//    val d = 5
//    val n = 1000
//    val weight = Array.fill(d){ Random.nextGaussian() }
//    val X = Array.fill(n,d){ Random.nextGaussian() }
//    val data = X.map( row =>
//      (if(row.zipAll(weight, 0.0, 0.0).map(x => x._1*x._2).sum > 0) 1.0 else 0.0) +: (row ++ Array("stuff", "", "another")))
//    val corruptedData =
//      data.map(row => row.map(x =>
//        if(math.random < 0.01)
//          (if(math.random < 0.5) "bug" else "")
//        else x.toString).mkString("\t") )
//
//    corruptedData
//
//  }
//
//
//  val rawData = makeToyLRData()
//
//
//  test("Created successfully and not numeric") {
//    sc = new spark.SparkContext("local[4]", "test")
//    val mc = new MLContext(sc)
//    val mlt = mc.loadStringArray(rawData)
//
//    assert(true)//!mlt.isNumeric)
//  }
//
//  test("Drop non-numeric") {
//    sc = new spark.SparkContext("local[4]", "test")
//    val mc = new MLContext(sc)
//    val mlt = mc.loadStringArray(rawData)
//
//    val mltNumeric = mlt.dropNonNumeric()
//    assert(mltNumeric.isNumeric)
//  }
//
//  test("Normalization/Transformation") {
//    sc = new spark.SparkContext("local[4]", "test")
//    val mc = new MLContext(sc)
//    val mlt = mc.loadStringArray(rawData)
//
//    val mltNumeric = mlt.dropNonNumeric()
//    val (normalizedMlt, transform) = mltNumeric.normalize(1 to mltNumeric.numCols)
//    val eps = 0.0001
//
//    assert(normalizedMlt.mean.reduce(_+_)/normalizedMlt.mean.length - 0.0 < eps)
//    assert(normalizedMlt.stdev.reduce(_+_)/normalizedMlt.stdev.length - 1.0 < eps)
//  }
//}
