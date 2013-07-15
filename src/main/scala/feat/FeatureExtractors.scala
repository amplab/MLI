package mli.feat

import mli.interface.MLTable

/**
 * Abstract Class to Support Feature Extraction. A Feature extractor is an object that expects a table as input
 * and produces a table as output. A typical feature processing pipeline will begin with raw data and end up with
 * numeric features.
 *
 */
abstract class FeatureExtractor {
  /**
   * Entry point for extraction.
   * @param in Input table.
   * @return Output table of extracted features.
   */
  def extract(in: MLTable): MLTable
  def apply(in: MLTable) = extract(in)
}

