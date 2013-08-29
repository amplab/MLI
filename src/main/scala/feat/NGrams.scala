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
  def buildDictionary(table: MLTable, col: Int, numGrams: Int, numFeatures: Int, stopWords: Set[String]): Seq[(String,Int)] = {

    //This is the classic Word Count, sorted descending by frequency.
    val freqTable = table.flatMap(rowToNgramRows(_, col, numGrams, stopWords))
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
  def rowToNgramRows(row: MLRow, col: Int, n: Int, stopWords: Set[String]): Seq[MLRow] = {
    val grams = ngrams(row(col).toString, n, stopWords)
    grams.map(s => MLRow(MLValue(s))).toSeq
  }

  //The main map function - given a row and a dictionary, return a new row which is an n-gram feature vector.
  def rowToFilteredNgrams(row: MLRow, col: Int, dict: Seq[(String,Int)], n: Int, stopWords: Set[String]): MLRow = {

    //Pull out the ngrams for a specific string.
    val x = ngrams(row(col).toString, n, stopWords)

    //Given my dictionary, create an Index,Value pair that indicates whether or not
    //the ngram was present in the string.
    val coll = dict.zipWithIndex.filter{ case((a,b), c) => x.contains(a)}.map {case ((a,b), c) => (c, MLValue(1.0))}

    //Return a new sparse row based on this feature vector.
    val sparsemlval = SparseMLRow.fromNumericSeq(row.drop(Seq(col)))
    SparseMLRow.fromNumericSeq(row.drop(Seq(col))) ++ SparseMLRow.fromSparseCollection(coll, dict.length, MLValue(0.0))
  }

  /**
   * Extract N-Grams
   * @param in Input corpus - MLTable.
   * @param c Input column - The text column you want transformed to Ngrams.
   * @param n Number of grams to compute (default: 1)
   * @param k Top-k features to keep.
   * @return Table of featurized data.
   */
  def extractNGrams(in: MLTable, c: Int=0, n: Int=1, k: Int=20000, stopWords: Set[String] = stopWords): (MLTable, MLRow => MLRow) = {

    //Build dictionary.
    val dict = buildDictionary(in, c, n, k, stopWords)

    //Extract the ngrams and pack into collection of rows.
    def featurizer(x: MLRow) = rowToFilteredNgrams(x, c, dict, n, stopWords)

    val out = in.map(featurizer(_))
    val existingColNames = in.drop(Seq(c)).schema.columns.map(_.name.getOrElse(""))

    //Set the column names of the table to match the NGram features.
    out.setColNames(existingColNames ++ dict.map(_._1))
    (out, featurizer)
  }

  /**
   * Extract TF-IDF score for each feature in an "N-Grammed" MLTable.
   * @param in MLTable of NGram features (result of extractNGrams)
   * @return TF-IDF features for an MLTable.
   */
  def tfIdf(in: MLTable, c: Int=0): MLTable = {
    val df = in.reduce(_ plus _)
    val df2 = df.toDoubleArray
    df2(c) = 1.0
    val df3 = MLVector(df2)
    val newtab = in.map(r => r over df3)
    newtab.setSchema(in.schema)
    newtab
  }

  /**
   * Default entry point.
   * @param in Input corpus.
   * @return Table of featurized data. Note, if you want to featurize new data in the same way, you'll need to call
   *         extractNGrams directly.
   */
  def extract(in: MLTable): MLTable = extractNGrams(in)._1


  /**
   * A list of common English stop words.
   */
  val stopWords = Set("a", "about", "above", "above", "across", "after",
    "afterwards", "again", "against", "all", "almost",
    "alone", "along", "already", "also","although","always",
    "am","among", "amongst", "amongst", "amount",  "an", "and",
    "another", "any","anyhow","anyone","anything","anyway", "anywhere",
    "are", "around", "as",  "at", "back","be","became", "because",
    "become","becomes", "becoming", "been", "before", "beforehand",
    "behind", "being", "below", "beside", "besides", "between", "beyond",
    "bill", "both", "bottom","but", "by", "call", "can", "cannot",
    "cant", "co", "con", "could", "couldnt", "cry", "de", "describe",
    "detail", "do", "done", "down", "due", "during", "each", "eg",
    "eight", "either", "eleven","else", "elsewhere", "empty",
    "enough", "etc", "even", "ever", "every", "everyone", "everything",
    "everywhere", "except", "few", "fifteen", "fify", "fill", "find",
    "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give",
    "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here",
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
    "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc",
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last",
    "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither", "never",
    "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not",
    "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one",
    "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves",
    "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re",
    "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she",
    "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
    "therein", "thereupon", "these", "they", "thick", "thin", "third", "this",
    "those", "though", "three", "through", "throughout", "thru", "thus", "to",
    "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un",
    "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well",
    "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which",
    "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves")
}