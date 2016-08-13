package com.posco.deeplearning.mlptest;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import com.beust.jcommander.Parameter;
import com.posco.deeplearning.util.IdxReader;
import com.posco.deeplearning.util.ModelUtils;


/**
 * @author Adam Gibson
 */
public class SparkPredictMLP { 

	@Parameter(names="-dataSavePath", description = "Directory in which to save the serialized data sets - required", required = true)
    private String dataSavePath = "/home/ec2-user/sparkdata/predict" + "/train-images.idx3-ubyte"; 
    private String labelSavePath = "/home/ec2-user/sparkdata/predict" + "/train-labels.idx1-ubyte";
    private String labeledPath = "/home/ec2-user/sparkdata/predict/img";
    private String outputPath = "/home/ec2-user/sparkdata/predict/img/" ;
    @Parameter(names="-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;
    private byte[][] imagesArr;
    private boolean testCodes = false;
    private boolean createImage = false;
    
    public SparkPredictMLP()
    {
        
    }

    public String neuralNet(String fileName) throws  Exception
    {
        
        //Model Loading 
    	System.out.println("[SparkPredictMLP][1] Load Trained Model Data");
        MultiLayerNetwork model = ModelUtils.loadModelAndParameters("/home/ec2-user/sparkdata/model", "MLPModel");

        //==image transforming function== 
        if(createImage)
        {
        	IdxReader.convert(dataSavePath, labelSavePath, outputPath);
        }
        
        //Default Test Code Use
        if(testCodes)
        {
        	DataSetIterator iter = new MnistDataSetIterator(64, true, 12345);
        	DataSet next = iter.next();
        }
        
        System.out.println("[SparkPredictMLP][2] Run Model ");
        //predict result
        INDArray features = Nd4j.create(IdxReader.convert2byte(dataSavePath, labelSavePath, outputPath,fileName));
        INDArray ia = model.output(features);
        INDArray guessRow = ia.getRow(0);  
      
        //evaluation
        StringBuffer resultSummary = new StringBuffer();
        resultSummary.append("==========================================</br>");
        resultSummary.append("[NeuralNetwork Result for Image Name : " + fileName + "]</br>" );
        resultSummary.append("==========================================</br>");
        double max = 0.0;
        int result = 0 ;
        for( int j = 0 ; j < 10 ; j++)
        {
        	System.out.println("[SparkPredictMLP][4] Result Decision " + guessRow.getDouble(j));
        	resultSummary.append("[" + j + "] : " + guessRow.getDouble(j) + "</br>" );
        	if(guessRow.getDouble(j) > max)
        	{
        		result = j;
        		max = guessRow.getDouble(j);
        	}
        }
        resultSummary.append("==========================================</br>");
        resultSummary.append("<b>Final Result : " + result + "</b></br>" );
        resultSummary.append("==========================================</br>");
        System.out.println("[SparkPredictMLP][5] Final Result " + result);
        
        //manage text
        
        
        return resultSummary.toString() + "";
    }

}
