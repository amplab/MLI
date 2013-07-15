package mli.interface

/**
 * Base class for basic ML types.
 */
abstract class MLValue() {
  def isEmpty: Boolean = false
  def isNumeric: Boolean
  def toNumber: Double
}

case class MLInt(value: Option[Int]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = true
  def toNumber = value.getOrElse(0).toDouble
}

case class MLDouble(value: Option[Double]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = true
  def toNumber = value.getOrElse(0.0)
}

case class MLString(value: Option[String]) extends MLValue {
  override def isEmpty = value.isEmpty
  def isNumeric = false
  def toNumber = 0.0
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

  implicit def doubleToMLValue(value: Double): MLValue = MLDouble(Option(value))
  implicit def stringToMLValue(value: String): MLValue = MLString(Option(value))
  implicit def intToMLValue(value: Int): MLValue = MLInt(Option(value))

  //Do we need an implicit for none?
  //implicit def emptyToMLValue(value: )
}