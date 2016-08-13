package com.posco.deeplearning.util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import org.deeplearning4j.nn.api.Updater;

/**
 * Project utility class to save and load models and parameters.
 */
public class ModelUtils {

    private static final Logger log = LoggerFactory.getLogger(ModelUtils.class);
    private ModelUtils(){}

    public static void saveModelAndParameters(MultiLayerNetwork net, String basePath, String fileName) {
        String confPath = FilenameUtils.concat(basePath, fileName+"-conf.json");
        String paramPath = FilenameUtils.concat(basePath, fileName + ".bin");
        String updaterPath = FilenameUtils.concat(basePath, fileName + "-updater.bin");
        log.info("Saving model and parameters to {} and {} ...",  confPath, paramPath);

        // save parameters
        try
        {
            File directory = new File(String.valueOf(basePath));

            if(!directory.exists())
            {
                directory.mkdir();
            }

            DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
            Nd4j.write(net.params(), dos);
            dos.flush();
            dos.close();

            // save model configuration
            //FileUtils.write(new File(confPath), net.conf().toJson());
            FileUtils.write(new File(confPath), net.getLayerWiseConfigurations().toJson());


            //Updater 업로드
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(updaterPath));
            oos.writeObject(net.getUpdater());
            oos.flush();
            oos.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    
    public static void saveSparkModelAndParameters(MultiLayerNetwork net, String basePath, String fileName) {
        String confPath = FilenameUtils.concat(basePath, fileName+"-conf.json");
        String paramPath = FilenameUtils.concat(basePath, fileName + ".bin");
        String updaterPath = FilenameUtils.concat(basePath, fileName + "-updater.bin");
        log.info("Saving model and parameters to {} and {} ...",  confPath, paramPath);
        
        

        // save parameters
        try
        {
            File directory = new File(String.valueOf(basePath));

            if(!directory.exists())
            {
                directory.mkdir();
            }

            DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
            Nd4j.write(net.params(), dos);
            dos.flush();
            dos.close();

            // save model configuration
            //FileUtils.write(new File(confPath), net.conf().toJson());
            FileUtils.write(new File(confPath), net.getLayerWiseConfigurations().toJson());


            //Updater 업로드
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(updaterPath));
            oos.writeObject(net.getUpdater());
            oos.flush();
            oos.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MultiLayerNetwork loadModelAndParameters(String basePath, String fileName) {
        log.info("Loading saved model and parameters...");
        MultiLayerNetwork savedNetwork = null;
        String confPath = FilenameUtils.concat(basePath, fileName+"-conf.json");
        String paramPath = FilenameUtils.concat(basePath, fileName + ".bin");
        String updaterPath = FilenameUtils.concat(basePath, fileName + "-updater.bin");

        File directory = new File(String.valueOf(basePath));

        if(!directory.exists())
        {
            return null;
        }
        if(!new File(confPath).exists())
        {
            return null;
        }
        File confFile = new File(confPath);

        // load parameters
        try
        {
            //load saved files
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(confFile));
            DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
            INDArray newParams = Nd4j.read(dis);
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(updaterPath));
            Updater updater = (Updater)ois.readObject();
            dis.close();
            ois.close();

            // set model with saved info
            savedNetwork = new MultiLayerNetwork(confFromJson);  //set Network conf
            savedNetwork.init();
            savedNetwork.setParams(newParams);                   //set Network parm
            savedNetwork.setUpdater(updater);                    //set Network updater

        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return savedNetwork;
    }

    public static void saveLayerParameters(INDArray param, String paramPath)  {
        // save parameters for each layer
        log.info("Saving parameters to {} ...", paramPath);

        try {
            DataOutputStream dos = new DataOutputStream(new FileOutputStream(paramPath));
            Nd4j.write(param, dos);
            dos.flush();
            dos.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    public static Layer loadLayerParameters(Layer layer, String paramPath) {
        // load parameters for each layer
        String name = layer.conf().getLayer().getLayerName();
        log.info("Loading saved parameters for layer {} ...", name);

        try{
            DataInputStream dis = new DataInputStream(new FileInputStream(paramPath));
            INDArray param = Nd4j.read(dis);
            dis.close();
            layer.setParams(param);
        } catch(IOException e) {
            e.printStackTrace();
        }

        return layer;
    }

    public static void saveParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }

    public static void saveParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) {
        Layer layer;
        for(String layerId: layerIds) {
            layer = model.getLayer(layerId);
            if (!layer.paramTable().isEmpty()) {
                ModelUtils.saveLayerParameters(layer.params(), paramPaths.get(layerId));
            }
        }
    }
    public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, int[] layerIds, Map<Integer, String> paramPaths) {
        Layer layer;
        for(int layerId: layerIds) {
            layer = model.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return model;
    }

    public static MultiLayerNetwork loadParameters(MultiLayerNetwork model, String[] layerIds, Map<String, String> paramPaths) {
        Layer layer;
        for(String layerId: layerIds) {
            layer = model.getLayer(layerId);
            loadLayerParameters(layer, paramPaths.get(layerId));
        }
        return model;
    }

    public static  Map<Integer, String>  getIdParamPaths(MultiLayerNetwork model, String basePath, int[] layerIds){
        Map<Integer, String> paramPaths = new HashMap<>();
        for (int id : layerIds) {
            paramPaths.put(id, FilenameUtils.concat(basePath, id + ".bin"));
        }

        return paramPaths;
    }

    public static Map<String, String> getStringParamPaths(MultiLayerNetwork model, String basePath, String[] layerIds){
        Map<String, String> paramPaths = new HashMap<>();

        for (String name : layerIds) {
            paramPaths.put(name, FilenameUtils.concat(basePath, name + ".bin"));
        }

        return paramPaths;
    }

    public static String defineOutputDir(String modelType){
        String tmpDir = System.getProperty("java.io.tmpdir");
        String outputPath = File.separator + modelType + File.separator + "output";
        File dataDir = new File(tmpDir,outputPath);
        if (!dataDir.getParentFile().exists())
            dataDir.mkdirs();
        return dataDir.toString();

    }

}