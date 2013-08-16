package mli.interface

import mli.interface.impl.SparkMLTable

class MLContext(val sc: spark.SparkContext) {

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
}