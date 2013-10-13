package ml.tree.impurity

object Gini extends Impurity {

   def calculate(c0 : Double, c1 : Double): Double = {
     val total = c0 + c1
     val f0 = c0 / total
     val f1 = c1 / total
     1 - f0*f0 - f1*f1
   }

 }
