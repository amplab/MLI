package ml.tree

import org.apache.spark.SparkContext._
import org.apache.spark.{Logging, SparkContext}
import ml.tree.impurity.{Variance, Entropy, Gini}
import ml.tree.strategy.Strategy

import ml.tree.node.NodeModel
import org.apache.spark.rdd.RDD

import ml.tree.Metrics.{accuracyScore,meanSquaredError}

object TreeRunner extends Logging {
  val usage = """
    Usage: TreeRunner <master>[slices] --strategy <Classification,Regression> --trainDataDir path --testDataDir path [--maxDepth num] [--impurity <Gini,Entropy,Variance>] [--samplingFractionForSplitCalculation num]
  """

  def main(args: Array[String]) {

    if (args.length < 2) {
		  System.err.println(usage)
		  System.exit(1)
	  }

    /**START Experimental*/
    System.setProperty("spark.cores.max", "8")
    /**END Experimental*/
    val sc = new SparkContext(args(0), "Decision Tree Runner",
      System.getenv("SPARK_HOME"), Seq(System.getenv("SPARK_EXAMPLES_JAR")))


    val arglist = args.toList.drop(1)
    type OptionMap = Map[Symbol, Any]

    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map
        case "--strategy" :: string :: tail => nextOption(map ++ Map('strategy -> string), tail)
        case "--trainDataDir" :: string :: tail => nextOption(map ++ Map('trainDataDir -> string), tail)
        case "--testDataDir" :: string :: tail => nextOption(map ++ Map('testDataDir -> string), tail)
        case "--impurity" :: string :: tail => nextOption(map ++ Map('impurity -> string), tail)
        case "--maxDepth" :: string :: tail => nextOption(map ++ Map('maxDepth -> string), tail)
        case "--samplingFractionForSplitCalculation" :: string :: tail => nextOption(map ++ Map('samplingFractionForSplitCalculation -> string), tail)
        case string :: Nil =>  nextOption(map ++ Map('infile -> string), list.tail)
        case option :: tail => println("Unknown option "+option)
                               exit(1)
      }
    }
    val options = nextOption(Map(),arglist)
    println(options)
    //TODO: Add check for acceptable string inputs

    val trainData = TreeUtils.loadLabeledData(sc, options.get('trainDataDir).get.toString)
    val strategyStr = options.get('strategy).get.toString
    val impurityStr = options.getOrElse('impurity,"Gini").toString
    val impurity = {
    	impurityStr match {
    	  case "Gini" => Gini
    	  case "Entropy" => Entropy
    	  case "Variance" => Variance
    	}
    }
    val maxDepth = options.getOrElse('maxDepth,"1").toString.toInt
    val fraction = options.getOrElse('samplingFractionForSplitCalculation,"1.0").toString.toDouble

    val tree = DecisionTree.train(
      input = trainData,
      numSplitPredicates = 1000,
      strategy = new Strategy(strategyStr),
      impurity = impurity,
      maxDepth = maxDepth,
      fraction = fraction,
      sparkContext = sc)
    println(tree)
    //println("prediction = " + tree.get.predict(Array(1.0, 2.0)))

    val testData = TreeUtils.loadLabeledData(sc, options.get('testDataDir).get.toString)


    val testError = {
      strategyStr match {
        case "Classification" => accuracyScore(tree, testData)
        case "Regression" => meanSquaredError(tree, testData)
      }
    }
    print("error = " + testError)

  }


}
