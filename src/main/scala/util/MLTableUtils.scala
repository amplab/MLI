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

  /**
   * Takes an input table and splits it into train and test set.
   * @param x Input table.
   * @param percentTrain Percent of data that should go into the.
   * @return Two MLTables - training and testing.
   */
  def trainTest(x: MLTable, percentTrain: Double=0.9): (MLTable, MLTable) = {
    val trainSet = x.filter(r => percentTrain >= (r.hashCode.abs % 1000).toDouble / 1000 )
    val testSet = x.filter(r => percentTrain < (r.hashCode.abs % 1000).toDouble / 1000 )
    (trainSet, testSet)
  }


}
