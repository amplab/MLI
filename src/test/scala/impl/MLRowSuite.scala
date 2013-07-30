package mli.test.impl

import org.scalatest.FunSuite
import mli.interface._


class MLRowSuite extends FunSuite {

  test("DenseMLRow can be constructed") {
    val row: DenseMLRow = DenseMLRow(MLValue(0.0), MLValue("foo"))
  }

  test("SparseMLRow can be constructed") {
    val row: SparseMLRow = SparseMLRow.fromSparseCollection(Map(0 -> MLValue(0.0), 100000 -> MLValue("foo")), 200000, MLValue(0.0))
  }

  test("MLRow.chooseRepresentation() chooses a sparse representation for a sparse vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10000).map({value => if (value == 400) MLValue(2) else MLValue(0)}))
    assert(row.getClass === classOf[SparseMLRow])
  }

  test("MLRow.chooseRepresentation() chooses a dense representation for a dense vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10000).map(MLValue.apply))
    assert(row.getClass === classOf[DenseMLRow])
  }

  test("MLRow.chooseRepresentation() chooses a dense representation for a small vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10).map({value => MLValue.apply(0.0)}))
    assert(row.getClass === classOf[DenseMLRow])
  }

  test("Mapping a SparseMLRow produces a SparseMLRow") {
    val row: MLRow = SparseMLRow.fromSparseCollection(Map(1 -> MLValue(1), 100000 -> MLValue(2)), 200000, MLValue(0.0))
    val mappedRow: MLRow = row.map(value => MLValue(value.toNumber * 2))
    assert(row.getClass === classOf[SparseMLRow])
  }

  test("Mapping a DenseMLRow produces a DenseMLRow") {
    val row: MLRow = DenseMLRow.fromSeq((0 until 10000).map(MLValue.apply))
    val mappedRow: MLRow = row.map(value => MLValue(value.toNumber + 1))
    assert(row.getClass === classOf[DenseMLRow])
  }
}
