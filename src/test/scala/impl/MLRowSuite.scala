package mli.test.interface

import org.scalatest.FunSuite
import mli.interface._


class MLRowSuite extends FunSuite {

  test("DenseMLRow can be constructed") {
    val row: DenseMLRow = DenseMLRow(MLDouble(0.0), MLString("foo"))
  }

  test("SparseMLRow can be constructed") {
    val row: SparseMLRow = SparseMLRow.fromSparseCollection(Map(0 -> MLDouble(0.0), 100000 -> MLString("foo")), 200000, MLDouble(0.0))
  }

  test("MLRow.chooseRepresentation() chooses a sparse representation for a sparse vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10000).map({value => if (value == 400) MLDouble(2) else MLDouble(0)}))
    assert(row.getClass === classOf[SparseMLRow])
  }

  test("MLRow.chooseRepresentation() chooses a dense representation for a dense vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10000).map(MLInt.apply))
    assert(row.getClass === classOf[DenseMLRow])
  }

  test("MLRow.chooseRepresentation() chooses a dense representation for a small vector") {
    val row: MLRow = MLRow.chooseRepresentation((0 until 10).map({value => MLDouble.apply(0.0)}))
    assert(row.getClass === classOf[DenseMLRow])
  }

  test("Mapping a SparseMLRow produces a SparseMLRow") {
    val row: MLRow = SparseMLRow.fromSparseCollection(Map(1 -> MLDouble(1), 100000 -> MLDouble(2)), 200000, MLDouble(0.0))
    val mappedRow: MLRow = row.map(value => MLDouble(value.toNumber * 2))
    assert(row.getClass === classOf[SparseMLRow])
  }

  test("Mapping a DenseMLRow produces a DenseMLRow") {
    val row: MLRow = DenseMLRow.fromSeq((0 until 10000).map(MLInt.apply))
    val mappedRow: MLRow = row.map(value => MLDouble(value.toNumber + 1))
    assert(row.getClass === classOf[DenseMLRow])
  }
}
