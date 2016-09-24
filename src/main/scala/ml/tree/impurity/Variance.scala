package ml.tree.impurity

import javax.naming.OperationNotSupportedException

object Variance extends Impurity {
   def calculate(c0: Double, c1: Double): Double = throw new OperationNotSupportedException("Variance.calculate")
 }
