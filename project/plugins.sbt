resolvers += "Sonatype snapshots" at "http://oss.sonatype.org/content/repositories/snapshots/"

addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.1.0-SNAPSHOT")

addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "2.0.0")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.7.2")
