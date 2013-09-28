import AssemblyKeys._

assemblySettings

name := "MLI"

version := "1.0"

organization := "edu.berkeley.cs.amplab"

scalaVersion := "2.9.3"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.9.3" % "0.8.0-incubating",
  "org.apache.spark" % "spark-mllib_2.9.3" % "0.8.0-incubating",
  "org.scalatest" %% "scalatest" % "1.9.1" % "test"
)

resolvers ++= Seq(
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo",
  "Spray" at "http://repo.spray.cc"
)

