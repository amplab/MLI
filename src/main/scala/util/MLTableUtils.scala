package mli.interface

/**
 * A collection of useful utilities for operating on MLTable Objects
 */
object MLTableUtils {
  //TODO: Implement toNumeric(?)
  //def toNumeric(mlt: MLTable): MLNumericTable = ???

  def normalize(mlnt: MLTable): (MLTable, MLRow => MLRow) = {
    //Todo: we can do this in one pass over the data by keeping track of sum(x**2).
    val mean = (mlnt.reduce(_ plus _)) over (mlnt.numRows.toDouble)
    val sd = mlnt.map(x => (x minus mean) times (x minus mean)).reduce(_ plus _)

    def trans(x: MLRow): MLRow = (x minus mean) over sd
    val res = mlnt.map(trans)

    (res, trans)
  }

}
