package com.dlblog.test1;

import javax.ws.rs.GET;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;

import com.posco.deeplearning.mlptest.SparkPredictMLP;
import com.posco.deeplearning.mlptest.SparkTrainMLP;
import com.posco.deeplearning.mlptest.TestSpark;

// Plain old Java Object it does not extend as class or implements 
// an interface

// The class registers its methods for the HTTP GET request using the @GET annotation. 
// Using the @Produces annotation, it defines that it can deliver several MIME types,
// text, XML and HTML. 

// The browser requests per default the HTML MIME type.

//Sets the path to base URL + /hello
@Path("/demo")
public class Hello 
{
  // This method is called if HTML is request
  @Path("predict")
  @GET
  @Produces(MediaType.TEXT_HTML)
  public String predictMLP(@QueryParam("filename") String fileName) 
  {
	  String resultVal = "" ;
	  SparkPredictMLP net = new SparkPredictMLP();
	  try 
      {
		  resultVal = net.neuralNet(fileName);
      } catch (Exception e) {
		 // TODO Auto-generated catch blocks
		 e.printStackTrace();
	  }
    return "<html> " + "<title>" + resultVal + "</title>"
        + "<body>" + resultVal + "</body>" + "</html> ";
  }

  // This method is called if HTML is request
  @Path("train")
  @GET
  @Produces(MediaType.TEXT_HTML)
  public String trainMLP() 
  {
	  String resultVal = "" ;
	  SparkTrainMLP net = new SparkTrainMLP();
	  try 
	  {
		 resultVal = net.main(null);
	  } catch (Exception e) {
		 // TODO Auto-generated catch block
	     e.printStackTrace();
	  }
    return "<html> " + "<title>" + resultVal + "</title>"
        + "<body>" + resultVal + "</body>" + "</html> ";
  }
} 
