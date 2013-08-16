package mli.interface

/**
 * Base class for basic ML types.
 */
abstract class MLValue() extends Ordered[MLValue] with Serializable {
  def isEmpty: Boolean = false
  def isNumeric: Boolean
  def toNumber: Double
  def compare(that: MLValue): Int = {
    if(this.isNumeric && that.isNumeric) this.toNumber compare that.toNumber
    else this.toString compare that.toString
  }
}

case class MLInt(value: Option[Int]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = true
  def toNumber = value.getOrElse(0).toDouble

  override def toString = value match {
    case Some(x) => x.toString
    case None => "None"
  }
}


case class MLDouble(value: Option[Double]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = true
  def toNumber = value.getOrElse(0.0)

  override def toString = value match {
    case Some(x) => x.toString
    case None => "None"
  }
}


case class MLString(value: Option[String]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = false
  def toNumber = 0.0

  override def toString = value match {
    case Some(x) => x.toString
    case None => "None"
  }
}


object MLValue {
  def apply(exp: String): MLValue = {
    if(exp.isEmpty) MLDouble(None)
    else {
      try{ MLInt(Some(exp.toInt)) } catch {
        case _ => try {
          MLDouble(Some(exp.toDouble)) } catch {
          case _ => MLString(Some(exp))
        }
      }
    }
  }

  def apply(value: Double): MLValue = MLDouble(Option(value))
  def apply(value: Int): MLValue = MLInt(Option(value))
  def apply(value: None.type): MLValue = MLDouble(None)

  implicit def doubleToMLValue(value: Double): MLValue = MLDouble(Option(value))
  implicit def stringToMLValue(value: String): MLValue = MLString(Option(value))
  implicit def intToMLValue(value: Int): MLValue = MLInt(Option(value))

  implicit def mlValueToDouble(value: MLValue): Double = value.toNumber
  implicit def mlValueToInt(value: MLValue): Int = value.toNumber.toInt
  implicit def mlValueToString(value: MLValue): String = value.toString

  //Do we need an implicit for none?
  //implicit def emptyToMLValue(value: )
}