package mli.interface.impl

import mli.interface._
import mli.impl.DenseMLMatrix
import mli.interface.MLTypes._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint


class MLNumericTable(@transient protected var rdd: RDD[MLVector], inSchema: Option[Schema] = None) extends Serializable {
  lazy val numCols = rdd.first.size

  lazy val numRows = rdd.count

  var tableSchema = inSchema

  var cachedMat: Option[RDD[LocalMatrix]] = None
  /**
   * Return the current schema. Since schema is inferred, we wait to do this until it is explicitly asked for in the
   * common case.
   */
  def schema(): Schema = {
    tableSchema match {
      case Some(s) => s
      case None => Schema(rdd.first)
    }
  }

  /**
   * Return a new RDD containing only the elements that satisfy a predicate.
   */
  def filter(f: MLVector => Boolean): MLNumericTable = new MLNumericTable(rdd.filter(f), tableSchema)

  /**
   * Return the union of this RDD and another one. Any identical elements will appear multiple
   * times (use `.distinct()` to eliminate them). For now this only works to union a DenseSparkMLTable to another.
   */
  def union(other: MLTable): MLNumericTable = this
  def union(other: MLNumericTable): MLNumericTable = {
    assert(schema == other.schema)
    new MLNumericTable(rdd.union(other.rdd), Some(schema))
  }

  /**
   * Takes a set of columns and returns their complement in the current schema. Useful for projection, sort, join, etc.
   * @param cols
   * @param schema
   * @return
   */
  def nonCols(cols: Seq[Index], schema: Schema): Seq[Index] = schema.columns.indices.diff(cols)

  /**
   * Return the join of this table and another one. Any identical elements will appear multiple
   * For now this doesn't work.
   */
  def join(other: MLTable, cols: Seq[Index]): MLNumericTable = this


  /**
   * Return a new RDD by applying a function to all elements of this RDD. Schema is inferred from the results.
   * User is expected to provide a map function which produces elements of a consistent schema as output.
   */
  def map(f: MLVector => MLVector): MLNumericTable = {
    val newRdd = rdd.map(f)

    MLNumericTable.fromMLVectorRdd(newRdd)
  }


  /**
   *  Return a new RDD by first applying a function to all elements of this
   *  RDD, and then flattening the results.
   */
  def flatMap(f: MLVector => TraversableOnce[MLVector]): MLNumericTable = new MLNumericTable(rdd.flatMap(f))

  /**
   *  Return a value by applying a reduce function to every element of the table.
   */
  def reduce(f: (MLVector, MLVector) => MLVector): MLVector = rdd.reduce(f)

  /**
   *  Run a reduce on all values of the row, grouped by key.
   */
  def reduceBy(key: Seq[Index], f: (MLVector, MLVector) => MLVector): MLNumericTable = {
    //val notKey = nonCols(key, schema)
    val newRdd = rdd.map(r => (r(key), r)).reduceByKey(f).map(_._2)

    MLNumericTable.fromMLVectorRdd(newRdd)
  }

  def pairToRow(p: (MLVector, MLVector)) = {
    MLVector(p._1 ++ p._2)
  }

  /**
   * Creates a new MLTable based on the cached version of the RDD.
   */
  def cache() = new MLNumericTable(rdd.cache(), tableSchema)

  /**
   * Sort a table based on a key.
   */
  def sortBy(key: Seq[Index], ascending: Boolean = true): MLNumericTable = {
    //val notKey = nonCols(key, schema)
    val newRdd = rdd.map(r => (r(key), r)).sortByKey(ascending).map(_._2)
    new MLNumericTable(newRdd, tableSchema)
  }

  /**
   * Return a value by applying a function to all elements of this the table and then reducing them.
   */
  def mapReduce(f: MLVector => MLVector, sum: (MLVector, MLVector) => MLVector): MLVector = rdd.map(f).reduce(sum)


  /**
   * Return a new MLTable by applying a function to each partition of the table, treating that partition as a Matrix.
   * This enables block-wise computation with a more natural interface than vanilla mapPartitions.
   */
  def matrixBatchMap(f: LocalMatrix => LocalMatrix): MLNumericTable = {
    //    def matrixMap(i: Iterator[MLVector]): Iterator[MLVector] = {
    //      val mat = DenseMLMatrix.fromVecs(i.map(_.toVector).toSeq)
    //      val res = f(mat)
    //      res.toMLRows
    //    }
    def cachedMatrixMap(m: LocalMatrix): Iterator[MLVector] = f(m).toMLVectors

    def createMatrix(i: Iterator[MLVector]): Iterator[LocalMatrix] = Iterator(DenseMLMatrix.fromVecs(i.toSeq))


    cachedMat match {
      case None => {
        cachedMat = Some(rdd.mapPartitions(createMatrix(_)).cache())
        MLNumericTable.fromMLVectorRdd(cachedMat.get.flatMap(cachedMatrixMap(_)))
      }
      case Some(value) => MLNumericTable.fromMLVectorRdd(value.flatMap(cachedMatrixMap(_)))
    }

    //SparkMLTable.fromMLRowRdd(cachedMat.map(cachedMatrixMap(_)))

    //SparkMLTable.fromMLRowRdd(rdd.mapPartitions(matrixMap(_)))
  }


  /**
   * Reduce the ml table to a table over a specific set of columns.
   */
  def project(cols: Seq[Index]) = {
    //TODO - project should project schema as well.
    map((row: MLVector) => MLVector(cols.map(i => row(i)).toSeq))
  }


  def take(count: Int) = rdd.take(count)


  /**
   * Sample the rows of the base table uniformly, with or without replacement.
   *
   * @param fraction Fraction of the records to sample.
   * @param withReplacement Sample with or without replacement.
   * @param seed Seed to use for random sampling.
   * @return Subsampled MLTable.
   */
  def sample(fraction: Double, withReplacement: Boolean, seed: Int) = {
    val newRdd = rdd.sample(withReplacement, fraction, seed)
    new MLNumericTable(newRdd, tableSchema)
  }

  /**
   * We support a toRDD operation here for interoperability with spark.
   */
  def toRDD(targetCol: Index = 0): RDD[LabeledPoint] = {
    val othercols = nonCols(Seq(targetCol), schema)
    rdd.map(r => LabeledPoint(r(targetCol), r(othercols).toDoubleArray))
  }

  def toDoubleArrayRDD: RDD[Array[Double]] = rdd.map(r => r.toDoubleArray)

  def toMLVectorRdd(): RDD[MLVector] = rdd


  /**
   * Do not use unless for last resort.
   */

  def collect() = rdd.collect


  def print() = rdd.collect.foreach(row => println(row.mkString("\t")))

  def print(count: Int) = take(count).foreach(row => println(row.mkString("\t")))

  def toMLTable = SparkMLTable.fromMLRowRdd(rdd.map(MLRow(_)))
}

object MLNumericTable {
  def doubleArrayToMLVector(a: Array[Double]): MLVector = {
    MLVector(a)
  }

  def apply(rdd: RDD[Array[Double]]): MLNumericTable = {
    val mldRdd = rdd.map(row => MLVector(row))
    new MLNumericTable(mldRdd)
  }

  def fromMLVectorRdd(rdd: RDD[MLVector]): MLNumericTable = {
    new MLNumericTable(rdd)
  }

}