package mli.feat

import mli.interface.{MLTableLike, MLNumericTable, MLTable}

/**
 * Abstract Class to Support Feature Extraction. A Feature extractor is an object that expects a table as input
 * and produces a table as output. A typical feature processing pipeline will begin with raw data and end up with
 * numeric features.
 *
 * @tparam U Input table type. (Usually MLTable or MLNumericTable)
 * @tparam T Output table type. (Usually MLNumericTable or MLTable)
 */
trait FeatureExtractor[U ,T] {
  def extract(in: U): T
  def apply(in: U) = extract(in)
}

/**
 * These classes support the most common type of feature extractors.
 */
trait RawFeatureExtractor extends FeatureExtractor[MLTable, MLTable]
trait RawNumericFeatureExtractor extends FeatureExtractor[MLTable, MLNumericTable]
trait NumericFeatureExtractor extends FeatureExtractor[MLNumericTable, MLNumericTable]

