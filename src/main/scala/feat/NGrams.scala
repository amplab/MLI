package mli.feat

import mli.interface._
import scala.collection.immutable.HashMap

/**
* The N-Gram Feature extractor expects a corpus of documents (an MLTable with one string column per row)
* as input and produces a numeric table of the top `k` n-grams by frequency as output.
*/
object NGrams extends FeatureExtractor with Serializable {

  /**
   * Helper method used to build total frequency table of all n-grams in the corpus.
   */
  def buildDictionary(table: MLTable, numGrams: Int, numFeatures: Int, stopWords: Set[String]): Seq[(String,Int)] = {

    //This is the classic Word Count, sorted descending by frequency.
    val freqTable = table.flatMap(rowToNgramRows(_, numGrams, stopWords))
                        .map(row => MLRow(row(0),1))
                        .reduceBy(Seq(0), (x,y) => MLRow(x(0),x(1).toNumber+y(1).toNumber))  //Run a reduceByKey on the first column.
                        .sortBy(Seq(1), false)    //Run a sortby on the second column.
                        .take(numFeatures)   //Return the top rows.
                        .map(r => (r(0).toString, r(1).toNumber.toInt))
    freqTable
  }

  /**
   *The ngram functionality removes all non-alpha/0-9 characters. Of course, if these are meaningful for you you
   * can extend this class and override the behavior.
   */
  def normalize(raw: String) = raw.replaceAll("[^a-zA-Z0-9]+","").toLowerCase

  //Does the actual N-Gram extraction from a string.
  def ngrams(str: String, n: Int, stopWords: Set[String] = Set[String](), sep: String="_"): Set[String] = {
    //Lowercase and split on whitespace, and filter out empty tokens and stopwords.
    val toks = str.toLowerCase.replaceAll(sep,"").split("\\s+").map(normalize).filter(_ != "").filter(!stopWords.contains(_))

    //Produce the ngrams.
    //val ngrams = (for( i <- 1 to n) yield toks.sliding(i).map(p => p.mkString(sep))).flatMap(x => x)
    val ngrams =  toks.sliding(n).map(p => p.mkString(sep))

    ngrams.toSet
  }

  //Responsible for tokenizing data for dictionary calculation.
  def rowToNgramRows(row: MLRow, n: Int, stopWords: Set[String]): Seq[MLRow] = {
    val grams = ngrams(row(0).toString, n, stopWords)
    grams.map(s => MLRow(MLValue(s))).toSeq
  }

  //The main map function - given a row and a dictionary, return a new row which is an n-gram feature vector.
  def rowToFilteredNgrams(row: MLRow, dict: Seq[(String,Int)], n: Int, stopWords: Set[String]): MLRow = {

    //Pull out the ngrams for a specific string.
    val x = ngrams(row(0).toString, n, stopWords)

    //Given my dictionary, create an Index,Value pair that indicates whether or not
    //the ngram was present in the string.
    val coll = dict.zipWithIndex.filter{ case((a,b), c) => x.contains(a)}.map {case ((a,b), c) => (c, MLValue(1.0))}

    //Return a new sparse row based on this feature vector.
    SparseMLRow.fromSparseCollection(coll, dict.length, MLValue(0.0))
  }

  /**
   * Extract N-Grams
   * @param in Input corpus - MLTable with one string column per row.
   * @param n Number of grams to compute (default: 1)
   * @param k Top-k features to keep.
   * @return Table of featurized data.
   */
  def extractNGrams(in: MLTable, n: Int=1, k: Int=20000, stopWords: Set[String] = Set[String]()): MLTable = {

    //Build dictionary.
    val dict = buildDictionary(in, n, k, stopWords)

    //Extract the ngrams and pack into collection of rows.
    val out = in.map(rowToFilteredNgrams(_, dict, n, stopWords))

    //Set the column names of the table to match the NGram features.
    out.setColNames(dict.map(_._1))
    out
  }

  /**
   * Extract TF-IDF score for each feature in an "N-Grammed" MLTable.
   * @param in MLTable of NGram features (result of extractNGrams)
   * @return TF-IDF features for an MLTable.
   */
  def tfIdf(in: MLTable): MLTable = {
    val df = in.reduce(_ plus _)
    in.map(_ over df)
  }

  /**
   * Default entry point.
   * @param in Input corpus.
   * @return Table of featurized data.
   */
  def extract(in: MLTable): MLTable = extractNGrams(in)
}