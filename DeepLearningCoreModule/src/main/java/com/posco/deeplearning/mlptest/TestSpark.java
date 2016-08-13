package com.posco.deeplearning.mlptest;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.data.DataSetExportFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;
 
import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;


import javax.ws.rs.Path;


/*
 * cd hue/apps/spark/java
mvn -DskipTests clean package
 */
/*
 * 1.2 환경변수를 앞으로 계속 적용하는 방법

     

      1.1과 유사한데요. 모든 사용자가 사용하고 계속 사용할 수 있는 환경변수를 적용하고자 하시면 /etc/bash.bashrc  

      (bash 쉘만 해당, zsh, csh은 다른 파일에...)

      위 파일의 마지막에 export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64/ 를 작성해주시면 됩니다.

      바로 환경변수가 적용되지 않아요. 

   

      source /etc/bash.bashrc 



 */
//mvn -DskipTests -Dspark.version=1.6.1 clean package
//sudo apt-get install build-essential python-dev
//./spark-1.6.1-bin-hadoop2.6/bin/spark-submit \
//--master spark://master-url \
//--class SparkAppMain \
//target/spark-getting-started-1.0-SNAPSHOT.jar
//netstat -at | grep 7077
//./spark-submit \ --master spark://172.31.11.219:7077 \ --class com.posco.spark.SparkTestLoader\ target/home/ubuntu/workspace/DeepLearningCoreModule/target/DeepLearningCoreModule-0.4-rc0-SNAPSHOT.jar
/*
* ./bin/spark-submit \
--class com.posco.spark.SparkTestLoader \
--master spark://ec2-52-78-79-219.ap-northeast-2.compute.amazonaws.com:6066 \
--deploy-mode cluster \
--supervise \
--executor-memory 2G \
--total-executor-cores 1 \
/home/ubuntu/workspace/DeepLearningCoreModule/target/DeepLearningCoreModule-0.4-rc0-SNAPSHOT.jar \
1000
*/
/*
http://ec2-52-78-79-219.ap-northeast-2.compute.amazonaws.com:6066/v1/submissions/create --header "Content-Type:application/json;charset=UTF-8" --data '{
	  "action" : "CreateSubmissionRequest",
	  "appArgs" : [ "myAppArgument1" ],
	  "appResource" : "file:/home/ubuntu/workspace/DeepLearningCoreModule/target/DeepLearningCoreModule-0.4-rc0-SNAPSHOT.jar",
	  "clientSparkVersion" : "1.5.2",
	  "environmentVariables" : {
	    "SPARK_ENV_LOADED" : "1"
	  },
	  "mainClass" : "com.posco.spark.SparkTestLoader"
	}'
	*/

//https://spark-summit.org/2014/wp-content/uploads/2014/07/Spark-Job-Server-Easy-Spark-Job-Management-Chan-Chu.pdf

public class TestSpark {
	public TestSpark()
	{
		
	}
	
	public String main(String a )
	{
		JavaSparkContext sc = null;
		try
		{
	    	System.out.println("============-useSparkLocal ");
	        SparkConf sparkConf = new SparkConf().setAppName("TestSpark");
	                //.set("spark.driver.allowMultipleContexts", "true")
	        		//.set("spark.local.ip", "172.31.11.219")
	        	    //.set("spark.driver.host", "172.31.11.219"); 
//	                .set("spark.driver.host", ""); 
//	        	    .set("spark.fileserver.port", "51811") 
//	        	    .set("spark.broadcast.port", "51812") 
//	        	    .set("spark.replClassServer.port", "51813") 
//	        	    .set("spark.blockManager.port", "51814") 
//	        	    .set("spark.executor.port", "51815") ;
	
	        //SparkConf sparkConf = new SparkConf().setMaster("spark://ec2-52-78-79-219.ap-northeast-2.compute.amazonaws.com:7077").setAppName("My").set("spark.driver.allowMultipleContexts", "true");;
	    	//SparkConf sparkConf = new SparkConf().setMaster("spark://127.0.0.1:7077").setAppName("My").set("spark.driver.allowMultipleContexts", "true");;
	        //SparkConf sparkConf = new SparkConf().setMaster("spark://172.31.11.219:7077").setAppName("My").set("spark.driver.allowMultipleContexts", "true");;
	        //SparkConf sparkConf = new SparkConf().setMaster("spark://ip-172-31-11-219:7077").setAppName("My");;
	        //SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("My").set("spark.driver.allowMultipleContexts", "true");;
	        sc = new JavaSparkContext("spark://ip-172-31-11-219:7077", "TestSpark");
			System.out.println("============-useSparkLocal : ");
		
		}
		catch(Exception e)
		{
			System.out.println("=======e : " + e);
		}
		finally
		{
			sc.stop();
			if(sc != null)
			{
				return sc.master();
			}
			else
			{
				return "X";
			}
			
		}

		
	}

}
