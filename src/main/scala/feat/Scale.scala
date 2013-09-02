package mli.feat

import scala.math.sqrt

import mli.interface.{MLRow, MLVector, MLTable}

/**
 * The Scale feature extractor rescales features according to their standard deviation. This will preserve
 * sparsity structure in Sparse feature tables. Future versions may also recenter the data.
 */
object Scale extends FeatureExtractor with Serializable {

  /**
   * This function scales an input table by the standard deviation of the value of its columns.
   * Only works on numeric data.
   * @param x Numeric input table.
   * @param label Index of the label (will not be scaled).
   * @return Scaled MLTable.
   */
  def scale(x: MLTable, label: Int = 0, featurizer: MLRow => MLRow): (MLTable, MLRow => MLRow) = {
    //First we compute the variance - note this could be done in one pass over the data.
    val ssq = x.map(r => r times r).reduce(_ plus _) over x.numRows
    val sum = x.reduce(_ plus _) over x.numRows
    val sd = ssq minus (sum times sum)

    //Now we divide by the standard deviation vector. We do not scale values with no standard deviation.
    val n = sd.toArray.map(sqrt).map(i => if(i == 0.0) 1.0 else i)

    //We also do not scale the training label.
    n(label) = 1.0
    val sdn = MLVector(n)

    val newtab = x.map(_ over sdn)
    newtab.setSchema(x.schema)
    def newfeaturizer(r: MLRow) = featurizer(r) over sdn

    (newtab, newfeaturizer)
  }

  def extract(in: MLTable): MLTable = scale(in, featurizer = r => r)._1

}