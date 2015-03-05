package org.template.recommendation

import io.prediction.controller.PAlgorithm
import io.prediction.controller.Params
import io.prediction.data.storage.BiMap

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.{Rating => MLlibRating}
import org.apache.spark.mllib.recommendation.ALSModel

import grizzled.slf4j.Logger

case class ALSAlgorithmParams(
  rank: Int,
  numIterations: Int,
  lambda: Double,
  seed: Option[Long]) extends Params

class ALSAlgorithm(val ap: ALSAlgorithmParams)
  extends PAlgorithm[PreparedData, ALSModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): ALSModel = {
    // MLLib ALS cannot handle empty training data.
    require(!data.ratings.take(1).isEmpty,
      s"RDD[Rating] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    // Convert user and girl String IDs to Int index for MLlib
    val userStringIntMap = BiMap.stringInt(data.ratings.map(_.user))
    val girlStringIntMap = BiMap.stringInt(data.ratings.map(_.girl))
    val mllibRatings = data.ratings.map( r =>
      // MLlibRating requires integer index for user and girl
      MLlibRating(userStringIntMap(r.user), girlStringIntMap(r.girl), r.rating)
    )

    // seed for MLlib ALS
    val seed = ap.seed.getOrElse(System.nanoTime)

    // If you only have one type of implicit event (Eg. "view" event only),
    // replace ALS.train(...) with
    //val m = ALS.trainImplicit(
      //ratings = mllibRatings,
      //rank = ap.rank,
      //iterations = ap.numIterations,
      //lambda = ap.lambda,
      //blocks = -1,
      //alpha = 1.0,
      //seed = seed)

    val m = ALS.train(
      ratings = mllibRatings,
      rank = ap.rank,
      iterations = ap.numIterations,
      lambda = ap.lambda,
      blocks = -1,
      seed = seed)

    new ALSModel(
      rank = m.rank,
      userFeatures = m.userFeatures,
      productFeatures = m.productFeatures,
      userStringIntMap = userStringIntMap,
      girlStringIntMap = girlStringIntMap)
  }

  def predict(model: ALSModel, query: Query): PredictedResult = {
    // Convert String ID to Int index for Mllib
    model.userStringIntMap.get(query.user).map { userInt =>
      // create inverse view of girlStringIntMap
      val girlIntStringMap = model.girlStringIntMap.inverse
      // recommendProducts() returns Array[MLlibRating], which uses girl Int
      // index. Convert it to String ID for returning PredictedResult
      val girlScores = model.recommendProducts(userInt, query.num)
        .map (r => girlScore(girlIntStringMap(r.product), r.rating))
      new PredictedResult(girlScores)
    }.getOrElse{
      logger.info(s"No prediction for unknown user ${query.user}.")
      new PredictedResult(Array.empty)
    }
  }

}
