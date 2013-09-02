package mli.interface

import org.apache.spark
import mli.interface.impl.SparkMLTable

class MLContext(@transient val sc: spark.SparkContext) extends Serializable {

  /**
   *
   * @param path Input path.
   * @param sep Separator - default is "tab"
   * @param isNumeric Is all the data numeric? Answering true here will speed up load time.
   * @return An MLTable which contains the data in the input path.
   */
  def loadFile(path: String, sep: String = "\t", isNumeric: Boolean = false): MLTable = {
    def parsePoint(x: String, sep: String): Array[Double] = {
      x.split(sep.toArray).map(_.toDouble)
    }

    val rdd = sc.textFile(path)
    if (isNumeric) SparkMLTable(rdd.map(parsePoint(_,sep)))
    //TODO: Need to build the non-numeric case. Also need to pass in header info.
    else SparkMLTable.fromMLRowRdd(rdd.map(x => DenseMLRow.fromSeq(x.split(sep).map(MLValue(_)))))
    //else new DenseSparkMLTable(rdd.map(_.split(sep.toArray).map(str => MLValue(str.trim()))).map(MLRow.chooseRepresentation))
  }

  /**
   * Specialized loader for dense, CSV data.
   * @param path Input path.
   * @param isNumeric By default, we assume that CSV data is not numeric.
   * @return Returns an MLTable which contains the data in the input path.
   */
  def loadCsvFile(path: String, isNumeric: Boolean = false): MLTable = loadFile(path, ",", isNumeric)

  def loadText(path: String): MLTable = new SparkMLTable(sc.textFile(path).map((x: String) => MLRow(MLValue(x))),
    Some(new Schema(Seq(ColumnSpec(Some("string"), ColumnType.String)))))

  def load(data: Array[Array[Double]], numSlices: Int = 4) = {
    //val newRdd = sc.makeRDD(data.map(row => MLRow.chooseRepresentation(row.map(MLValue(_)))))
    val newRdd = sc.makeRDD(data)
    SparkMLTable(newRdd)
  }

  /**
   * Loads data in LibSVM or augmented LibSVM (index on first column) format.
   *
   * Expects data to look like:
   * 0:1.00 4:0.005 ... index:value
   *
   * Assumes indexes are monotonically increasing.
   * Current implementation takes one full pass over data to determine dimensionality.
   *
   * Todo - add support for index free first column.
   *
   * @param path
   * @return
   */
  def loadSparseFile(path: String): MLTable = {

    //Takes a line and returns a sequence of (index,value) pairs.
    def parseSparsePoint(text: String): Array[(Int, Double)] = {
      val items = text.split(" ")

      items.map(r => {
        val res = r.split(":")
        (res(0).toInt, res(1).toDouble)
      })
    }

    def makeRow(x: Array[(Int, Double)], maxIndex: Int): MLRow = {
      val newRow = x.map(v => (v._1, MLDouble(Some(v._2)))).toIterable
      SparseMLRow.fromSparseCollection(newRow, maxIndex, 0.0)
    }
    val rdd = sc.textFile(path).map(parseSparsePoint)

    val maxIndex = rdd.map(x => x.last._1).reduce(_ max _)

    val newRdd = rdd.map(makeRow(_, maxIndex))
    SparkMLTable.fromMLRowRdd(newRdd)
  }
}