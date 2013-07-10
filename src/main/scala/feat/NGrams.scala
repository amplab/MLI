//package mli.feat
//
//import mli.interface.{MLTableUtils, MLTable, MLNumericTable}
//import scala.collection.immutable.HashMap
//
///**
// * The N-Gram Feature extractor expects a corpus of documents (an MLTable with one string column per row)
// * as input and produces a numeric table of the top `k` n-grams as output.
// */
//object NGrams extends NumericFeatureExtractor with Serializable {
//  /**
//   * Helper method used to build total frequency table of all n-grams in the corpus.
//   * @param table
//   * @param num_features
//   * @return
//   */
//  def buildDictionary(table: MLTable,num_features:Int): HashMap[String,Int] = {
//    val word_list = table.map(word => (word,1)).reduceByKey(_ + _).map(count => new Tuple2(count._2,count._1)).sortByKey(false).collect()
//    val dictionary = new HashMap[String,(Int,Int)]()
//    for(i <- 0 until num_features) {
//      val key = word_list(i)._2 //stop word removal
//      dictionary += key -> (i,word_list(i)._1)
//    }
//    dictionary
//  }
//
//  /**
//   * Extract N-Grams
//   * @param in Input corpus - MLTable with one string column per row.
//   * @param n Number of grams to compute (default: 1)
//   * @param k Top-k features to keep.
//   * @return Table of featurized data.
//   */
//  def extractNGrams(in: MLTable, n: Int=1, k: Int=20000): MLNumericTable = {
//    assert(in.numCols == 1)
//
//    //Build dictionary.
//
//    //Build
//
//    MLTableUtils.toNumeric(in)
//  }
//
//  /**
//   * Default entry point.
//   * @param in Input corpus.
//   * @return Table of featurized data.
//   */
//  def extract(in: MLTable): MLNumericTable = extractNGrames(in)
//}