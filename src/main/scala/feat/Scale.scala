import scala.math.sqrt

import mli.feat.FeatureExtractor
import mli.interface.{MLVector, MLTable}

/**
 * The scale feature extractor rescales features according to their standard deviation. This will preserve
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
  def scale(x: MLTable, label: Int = 0): MLTable = {
    //First we compute the variance - note this could be done in one pass over the data.
    val ssq = x.map(r => r times r).reduce(_ plus _) over x.numRows
    val sum = x.reduce(_ plus _) over x.numRows
    val sd = ssq minus (sum times sum)

    //Now we divide by the standard deviation vector. We do not scale values with no standard deviation.
    val n = sd.toArray.map(sqrt).map(i => if(i == 0.0) 1.0 else i)

    //We also do not scale the training label.
    n(label) = 1.0
    val sdn = MLVector(n)

    x.map(_ over sdn)
  }

  def extract(in: MLTable): MLTable = scale(in)

}