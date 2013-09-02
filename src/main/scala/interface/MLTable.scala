package mli.interface

import mli.interface.impl.SparkMLTable
import mli.interface.MLTypes._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint


/**
 * Enumerated column type. Currently supports Int, Double, String, and Empty.
 */
object ColumnType extends Enumeration with Serializable{
  val Int, Double, String, Empty = Value
}


/**
 * Contains metadata about a particular column. Contains an Optional name and enumerated Column Type.
 * @param name Optional Column name currently accessed through Schema.lookup()
 * @param kind Enumerated Column type.
 */
class ColumnSpec(val name: Option[String], val kind: ColumnType.Value) extends Serializable
object ColumnSpec {
  def apply(name: Option[String], kind: ColumnType.Value): ColumnSpec = new ColumnSpec(name, kind)
}


/**
 * A schema represents the types of the columns of an MLTable. Users may use schema information to infer
 * properties of the table columns - which are numeric vs. text, which have missing values, etc.
 * @param columns The specification of each column, in order.
 */
class Schema(val columns: Seq[ColumnSpec]) extends Serializable {
  val hasText: Boolean = columns.map(_.kind).contains(ColumnType.String)

  val hasMissing: Boolean = columns.map(_.kind).contains(ColumnType.Empty)

  val isNumeric: Boolean = columns.forall(Set(ColumnType.Int, ColumnType.Double) contains _.kind)

  val numericCols: Seq[Index] = columns.zipWithIndex.filter(Set(ColumnType.Int, ColumnType.Double) contains _._1.kind).map(_._2)
  val emptyCols: Seq[Index] = columns.zipWithIndex.filter(_._1.kind == ColumnType.Empty).map(_._2)
  val textCols: Seq[Index] = columns.zipWithIndex.filter(_._1.kind == ColumnType.String).map(_._2)

  /**
   * Function
   * @param other
   * @param cols
   * @return
   */
  def join(other: Schema, cols: Seq[Index]): Schema = {

    val joincols = cols.map(columns(_))
    val otherjoincols = cols.map(other.columns(_))
    assert(joincols == otherjoincols)

    val t1OtherSchema = columns.indices.diff(cols).map(columns(_))
    val t2OtherSchema = other.columns.indices.diff(cols).map(other.columns(_))

    new Schema(joincols ++ t1OtherSchema ++ t2OtherSchema)
  }

  override def toString = columns.zipWithIndex.map { case (c,i) => c.name.getOrElse(i) }.mkString("\t")

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
 */
trait MLTable {
  val numCols: Int
  val numRows: Long
  var tableSchema: Option[Schema]

  def schema(): Schema

  def filter(f: MLRow => Boolean): MLTable
  def union(other: MLTable): MLTable
  def map(f: MLRow => MLRow): MLTable
  def mapReduce(m: MLRow => MLRow, r: (MLRow, MLRow) => MLRow ): MLRow
  def matrixBatchMap(f: MLMatrix => MLMatrix): MLTable
  def project(cols: Seq[Index]): MLTable
  def join(other: MLTable, cols: Seq[Index]): MLTable
  def flatMap(m: MLRow => TraversableOnce[MLRow]): MLTable
  def cache(): MLTable

  def reduce(f: (MLRow, MLRow) => MLRow): MLRow
  def reduceBy(keys: Seq[Index], f: (MLRow, MLRow) => MLRow): MLTable
  def sortBy(keys: Seq[Index], ascending: Boolean=true): MLTable
  //No support for full table to Matrix just yet.
  //def toMatrix: MLMatrix

  //No support for iterator yet.
  //def iterator(): Iterator[MLRow]
  def collect(): Seq[MLRow]
  def take(n: Int): Seq[MLRow]

  //We support toRDD to for interoperability with Spark.
  def toRDD(targetCol: Index = 0): RDD[LabeledPoint]

  //Concrete methods provided by the interface below.
  def project(cols: => Seq[String]): MLTable = {
    project(schema.lookup(cols))
  }

  def drop(cols: Seq[Index]) = {
    val converse = (0 until numCols).diff(cols).toArray
    project(converse)
  }

  def setSchema(newSchema: Schema) = {
    tableSchema = Some(newSchema)
  }

  override def toString = {
    schema.toString + "\n" + this.take(200).mkString("\n")
  }

  def setColNames(names: Seq[String]) = {
    val theSchema = schema()
    val newcols = (0 until names.length).map(i => new ColumnSpec(Some(names(i)), theSchema.columns(i).kind))
    tableSchema = Some(new Schema(newcols))
  }
}

object MLTable {
  def apply(dat: RDD[Array[Double]]) = SparkMLTable(dat)
}


