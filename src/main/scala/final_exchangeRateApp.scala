// load librarys
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

// define the class for Exchange datas
case class Exchange(label:String, time:String)
case class ResultForm(mean:String, raw:Array[Exchange])

object final_exchangeRateApp {
    def main(args:Array[String]) {
        // create SparkSession object
        val spark = SparkSession.builder.appName("exchangeRateApp").getOrCreate()

        // import libraries
        import spark.implicits._
        import org.apache.spark.sql.functions.explode

        // adjust the level of output messages for console
        val sc = spark.sparkContext
        sc.setLogLevel("WARN")

        // Get spark data frame from api (url: prividing daily exchange rate data)
        val result = scala.io.Source.fromURL("http://ecos.bok.or.kr/api/StatisticSearch/U8U9S9MEHH8NU1X1BRBV/json/kr/1/100000/036Y001/DD/20100101/20200628/0000001/?/?/").mkString
        //only one line inputs are accepted. (I tested it with a complex Json and it worked)
        val jsonResponseOneLine = result.toString().stripLineEnd
        //You need an RDD to read it with spark.read.json! This took me some time. However it seems obvious now
        val jsonRdd = spark.sparkContext.parallelize(jsonResponseOneLine :: Nil)

        // val jsonDf = spark.read.json(jsonRdd)  // statisticSearch 기준, row레벨것 따오기
        val jsonDf = spark.read.json(jsonRdd).select("StatisticSearch.row")
        var dataDF = jsonDf.select(explode(jsonDf("row"))).toDF("row").select("row.DATA_VALUE","row.TIME")
        
        // renaming the "DATA_VALUE" column into "label" for train
        dataDF = dataDF.withColumnRenamed("DATA_VALUE","label")

        // convert dataframe to dataset according to Exchange class
        var dataDS = dataDF.as[Exchange]

        // cache dataset
        dataDS.cache()
        dataDF.cache()

        // display the schema of dataset
        //dataDs.printSchema()

        // get only mean of KRW for the entire period
        val meanOfWhole = dataDS.describe("label").rdd.map{
r=>(r.getAs[String]("summary"),r.get(1))
}.filter(_._1 == "mean").map(_._2).collect


        // define transformation-Filter only if it is less than meanValue
        val cheapDS = dataDS.filter(dataDF("label") < meanOfWhole(0)).sort($"TIME".desc)


        import org.apache.spark.sql.functions.udf

        // define udf to make strings numeric
        val getYear = udf((s: String) => {
            val year = (s.substring(0,4)).toInt
            year
        })

        // define udf to make strings numeric
        val getMonth = udf((s: String) => {
            val month = (s.substring(4,6)).toInt
            month
        })

        // define udf to make strings numeric
        val getDate = udf((s: String) => {
            val date = (s.substring(6,8)).toInt
            date
        })

        // define udf to make strings numeric
        val castToDouble = udf[Double, String](_.toDouble)

        // add new columns for train as feature vector
        dataDF = dataDF.withColumn("year",getYear(dataDF("TIME")))
        dataDF = dataDF.withColumn("month",getMonth(dataDF("TIME")))
        dataDF = dataDF.withColumn("date",getDate(dataDF("TIME")))

        // type casting
        dataDF = dataDF.withColumn("label",castToDouble(dataDF("label")))

        import scala.math.{Pi,sin}

        // define udf for encoding month using sin frequency
        val encodeMonth = udf((_month: Int) => {
            val month = sin(Pi*_month/12)
            month
        })

        dataDF = dataDF.withColumn("encodedMonth",encodeMonth(dataDF("month")))

        import org.apache.spark.ml.feature.VectorAssembler

        // gathering all the features in on column called "Assembler"
        var assembler = new VectorAssembler().setInputCols(Array("year","encodedMonth","date")).setOutputCol("features")
        dataDF=assembler.transform(dataDF)

        // Splitting the data into training(0.8) and validation(0.2) with seed 42
        var Array(train, validation) = dataDF.randomSplit(Array(.8, .2) , 42)

        import org.apache.spark.ml.feature.StandardScaler

        // feature scaling
        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(true)

        // setting range for scaling only considering train data and trasforming
        var scaledTrainDF =  scaler.fit(train).transform(train)
        // applying only transformations to validation data
        var scaledValidationDF =  scaler.fit(train).transform(validation)

        import org.apache.spark.ml.feature.Normalizer

        // normalizing via L2 norm method to prevent overfiiting for train data
        var normalizedScaledTrainDF = new Normalizer().setInputCol("scaledFeatures").setOutputCol("normFeatures").setP(2.0).transform(scaledTrainDF)

        // also normalizing via L2 norm method to prevent overfiiting for validation data
        var normalizedScaledValidationDF = new Normalizer().setInputCol("scaledFeatures").setOutputCol("normFeatures").setP(2.0).transform(scaledValidationDF)

        import org.apache.spark.ml.regression.LinearRegression

        // fitting train data on linear regressor, I consider types of regularization I want to use, Lasso method will provide indirect functions of selecting variables, so I will set elastic net param as 1.0 in order to activate Lasso method
        var lr = new LinearRegression().setFeaturesCol("normFeatures").setMaxIter(10).setRegParam(1.0).setElasticNetParam(1.0)
        var lrModel = lr.fit(normalizedScaledTrainDF)

        // 20200629 테스트
        var newdata=Seq(Exchange("0","20200629")).toDF()

        // 학습된 모델에 테스트 시행
        // add new columns for train as feature vector
        newdata = newdata.withColumn("year",getYear(newdata("TIME")))
        newdata = newdata.withColumn("month",getMonth(newdata("TIME")))
        newdata = newdata.withColumn("date",getDate(newdata("TIME")))

        // type casting
        newdata = newdata.withColumn("label",castToDouble(newdata("label")))

        // encode month data
        newdata = newdata.withColumn("encodedMonth",encodeMonth(newdata("month")))

        // gathering all the features in on column called "Assembler"
        var assembler2 = new VectorAssembler().setInputCols(Array("year","encodedMonth","date")).setOutputCol("features")
        newdata=assembler2.transform(newdata)

        // applying only transformations to validation data
        var scaledNewDF =  scaler.fit(train).transform(newdata)

        // also normalizing via L2 norm method to prevent overfiiting for validation data
        var normalizedScaledNewDF = new Normalizer().setInputCol("scaledFeatures").setOutputCol("normFeatures").setP(2.0).transform(scaledNewDF)

        // predict
        var lrPredictions_test = lrModel.transform(normalizedScaledNewDF)

        // create new dataframe for converting to json
        val rst = Seq(ResultForm(meanOfWhole(0).asInstanceOf[String],cheapDS.collect))
        val rst_df = rst.toDF

        // put on console
        println("Prediction : ")
        lrPredictions_test.select($"TIME",$"prediction").show()
        println("Filtering only if it is less than mean value of whole period : ")
        cheapDS.show(50)
        println("Serialize result data : ")
        rst_df.show()

        // save json file
        //rst_df.write.format("json")
        rst_df.write.format("json").save("/sparkdata/exchangerate/output5")

        // turn orr the dataset caching
        dataDS.unpersist()
        dataDF.unpersist()
    }
}



