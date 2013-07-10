package mli.interface

import mli.interface.impl.DenseSparkMLTable
import mli.interface.MLTypes._

import spark.SparkContext
import SparkContext._

// Base type for all MLBase types (like Object, generic)
abstract class MLValue() {
  def isEmpty: Boolean = false
  def isNumeric: Boolean
  def toNumber: Double
}

case class MLEmpty() extends MLValue {
  override def isEmpty = true
  override def isNumeric = false
  def toNumber = 0.0
}

case class MLInt(value: Int) extends MLValue {
  def isNumeric = true
  def toNumber = value.toDouble
}

case class MLDouble(value: Double) extends MLValue {
  def isNumeric = true
  def toNumber = value
}

case class MLString(value: String) extends MLValue {
  def isNumeric = false
  def toNumber = 0.0
}

object MLValue {
  val empty = new MLEmpty()
  def apply(exp: String): MLValue = {
    if(exp.isEmpty) empty
    else {
      try{ MLInt(exp.toInt) } catch {
        case _ => try {
          MLDouble(exp.toDouble) } catch {
          case _ => MLString(exp)
        }
      }
    }
  }
  def apply(value: Double): MLValue = MLDouble(value)
  def apply(value: Int): MLValue = MLInt(value)

  implicit def doubleToMLValue(value: Double): MLValue = MLDouble(value)
  implicit def stringToMLValue(value: String): MLValue = MLString(value)
  implicit def intToMLValue(value: Int): MLValue = MLInt(value)

  //Do we need an implicit for none?
  //implicit def emptyToMLValue(value: )
}

object ColumnType extends Enumeration with Serializable{
  val Int, Double, String, Empty = Value
}

class ColumnSpec(val name: Option[String], val kind: ColumnType.Value) extends Serializable

class Schema(val columns: Seq[ColumnSpec]) extends Serializable {
  lazy val hasText: Boolean = columns.map(_.kind).contains(ColumnType.String)

  lazy val hasMissing: Boolean = columns.map(_.kind).contains(ColumnType.Empty)

  lazy val isNumeric: Boolean = columns.forall(Set(ColumnType.Int, ColumnType.Double) contains _.kind)

  lazy val numericCols: Seq[Index] = columns.zipWithIndex.filter(Set(ColumnType.Int, ColumnType.Double) contains _._1.kind).map(_._2)
  lazy val emptyCols: Seq[Index] = columns.zipWithIndex.filter(_._1.kind == ColumnType.Empty).map(_._2)
  lazy val textCols: Seq[Index] = columns.zipWithIndex.filter(_._1.kind == ColumnType.String).map(_._2)

  //Helper functions.

  /**
   * Return column indexes from column names. Current implementation is expensive for wide rows.
   * @param names Column names of interest.
   * @return A list of indexes in order corresponding to the string names.
   *         If a column name does not exist, it is omitted from the result list.
   */
  def lookup(names: Seq[String]): Seq[Index] = names.map(n => columns.indexWhere(c => c.name.getOrElse("") == n)).filter(_ != -1)
}

object Schema {
  def apply(row: MLRow) = new Schema(row.map(c => {
    c match {
      case MLEmpty() => new ColumnSpec(None, ColumnType.Empty)
      case MLInt(i) => new ColumnSpec(None, ColumnType.Int)
      case MLDouble(d) => new ColumnSpec(None, ColumnType.Double)
      case MLString(s) => new ColumnSpec(None, ColumnType.String)
    }
  }))
}

class SchemaException(val error: String) extends Exception


/**
 * This is the base interface for an MLTable object and defines the basic operations it needs to support.
 * All MLTables must have a Schema, which defines their column structure, as well as a fixed number of rows and columns.
 * The additional operations they support are a combination of traditional relational operators and MapReduce primitives
 * designed to give a developer a familiar interface to the
 * @tparam U
 */
trait MLTableLike[U] {
  val numCols: Int
  val numRows: Long
  val schema: Schema

  def filter(f: U => Boolean): MLTableLike[U]
  def union(other: MLTableLike[U]): MLTableLike[U]
  def map(f: U => U): MLTableLike[U]
  def mapReduce(m: U => U, r: (U,U) => U ): U
  def matrixBatchMap(f: MLMatrix => MLMatrix): MLTableLike[U]
  def project(cols: Seq[Index]): MLTableLike[U]
  def join(other: MLTableLike[U], cols: Seq[Index]): MLTableLike[U]
  def flatMap(m: U => TraversableOnce[U]): MLTableLike[U]

  def reduce(f: (U,U) => U): U
  //No support for full table to Matrix just yet.
  //def toMatrix: MLMatrix

  //No support for iterator yet.
  //def iterator(): Iterator[MLRow]

  //Concrete methods provided by the interface below.
  def project(cols: => Seq[String]): MLTableLike[U] = {
    project(schema.lookup(cols))
  }
}

abstract class MLTable extends MLTableLike[MLRow]
abstract class MLNumericTable extends MLTableLike[MLVector]


object MLTable {
  def apply(dat: spark.RDD[Array[Double]]) = DenseSparkMLTable(dat)
}


class MLContext(val sc: spark.SparkContext) {

  /**
   *
   * @param path Input path.
   * @param sep Separator - default is "tab"
   * @param isNumeric Is all the data numeric? Answering true here will speed up load time.
   * @return An MLTable which contains the data in the input path.
   */
  def loadFile(path: String, sep: String = "\t", isNumeric: Boolean = false): MLNumericTable = {
    def parsePoint(x: String, sep: String): Array[Double] = {
      x.split(sep.toArray).map(_.toDouble)
    }

    val rdd = sc.textFile(path)
    if (isNumeric) DenseSparkMLTable(rdd.map(parsePoint(_,sep)))
    //TODO: Need to build the non-numeric case. Also need to pass in header info.
    else DenseSparkMLTable(rdd.map(parsePoint(_,sep)))
    //else new DenseSparkMLTable(rdd.map(_.split(sep.toArray).map(str => MLValue(str.trim()))).map(MLRow.chooseRepresentation))
  }

  /**
   * Specialized loader for dense, CSV data.
   *@param path Input path.
   * @param isNumeric By default, we assume that CSV data is not numeric.
   * @return Returns an MLTable which contains the data in the input path.
   */
  def loadCsvFile(path: String, isNumeric: Boolean = false): MLNumericTable = loadFile(path, ",", isNumeric)

  def load(data: Array[Array[Double]], numSlices: Int = 4) = {
    new DenseSparkMLTable(sc.makeRDD(data.map(MLVector(_))))
  }
}