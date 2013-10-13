package ml.tree.impurity

object Entropy extends Impurity {

   def log2(x: Double) = scala.math.log(x) / scala.math.log(2)

   def calculate(c0: Double, c1: Double): Double = {
     if (c0 == 0 || c1 == 0) {
       0
     } else {
       val total = c0 + c1
       val f0 = c0 / total
       val f1 = c1 / total
       -(f0 * log2(f0)) - (f1 * log2(f1))
     }
   }

 }
