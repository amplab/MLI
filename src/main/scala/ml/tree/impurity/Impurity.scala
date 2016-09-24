package ml.tree.impurity

import ml.tree.node.NodeStats
import ml.tree.split.Split

trait Impurity {

  def calculateClassificationGain(split: Split, calculations : Map[(Split, String, Double),Long]): Double = {
          val leftRddZeroCount = calculations.getOrElse((split,"left",0.0),0L).toDouble;
          val rightRddZeroCount = calculations.getOrElse((split,"right",0.0),0L).toDouble;
          val leftRddOneCount = calculations.getOrElse((split,"left",1.0),0L).toDouble;
          val rightRddOneCount = calculations.getOrElse((split,"right",1.0),0L).toDouble;
          val leftRddCount = leftRddZeroCount + leftRddOneCount;
          val rightRddCount = rightRddZeroCount + rightRddOneCount;
          val totalZeroCount = leftRddZeroCount + rightRddZeroCount;
          val totalOneCount = leftRddOneCount + rightRddOneCount;
          val totalCount = totalZeroCount + totalOneCount;
          val gain = {
            if (leftRddCount == 0 || rightRddCount == 0) 0
            else {
              val topGini = calculate(totalZeroCount,totalOneCount)
              val leftWeight = leftRddCount / totalCount
              val leftGini = calculate(leftRddZeroCount,leftRddOneCount) * leftWeight
              val rightWeight = rightRddCount / totalCount
              val rightGini = calculate(rightRddZeroCount,rightRddOneCount) * rightWeight
              topGini - leftGini - rightGini
            }
          }
          gain
  }

  def calculateRegressionGain(split: Split, calculations : Map[(Split, String),(Double, Double, Long)], nodeStats : NodeStats): (Double, NodeStats, NodeStats) = {
    val topCount = nodeStats.count.get
    val leftCount = calculations.getOrElse((split,"left"),(0,0,0L))._3
    val rightCount = calculations.getOrElse((split,"right"),(0,0,0L))._3
    if (leftCount == 0 || rightCount == 0){
    	// No gain return values
    	//println("leftCount = " + leftCount + "rightCount = " + rightCount + " topCount = " + topCount)
          (0, new NodeStats, new NodeStats)
    } else{
    	val topVariance = nodeStats.variance.get
    	val leftMean = calculations((split,"left"))._1
    	val leftVariance = calculations((split,"left"))._2
    	val rightMean = calculations((split,"right"))._1
    	val rightVariance = calculations((split,"right"))._2
    	//TODO: Check and if needed improve these toDouble conversions
    	val gain = topVariance - ((leftCount.toDouble / topCount) * leftVariance) - ((rightCount.toDouble/topCount) * rightVariance)
    	(gain,
    	    new NodeStats(mean = Some(leftMean), variance = Some(leftVariance), count = Some(leftCount)),
    	    new NodeStats(mean = Some(rightMean), variance = Some(rightVariance), count = Some(rightCount)))
    }
  }

  def calculate(c0 : Double, c1 : Double): Double

}
