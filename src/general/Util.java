package general;

import java.io.*;

import id3.MyID3;
import j48.MyJ48;
import weka.classifiers.trees.Id3;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class utility
 * 
 * @author rikysamuel
 */
public class Util {
    
    private static Instances data;
    private static Classifier classifier;

    public static Instances getData() {
        return data;
    }
    
    public static Classifier getClassifier() {
        return classifier;
    }
    
    /**
     * load dataset from ARFF format
     * @param filename 
     */
    public static void loadARFF(String filename) {
        FileReader file = null;
        try {
            file = new FileReader(filename);
            try (BufferedReader reader = new BufferedReader(file)) {
                data = new Instances(reader);
            }
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                if (file!=null) {
                    file.close();
                }
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * load dataset from CSV format
     * @param filename 
     */
    public static void loadCSV(String filename) {
        try {
            CSVLoader csv = new CSVLoader();
            csv.setFile(new File(filename));
            data = csv.getDataSet();
            
            // setting class attribute
            data.setClassIndex(data.numAttributes() - 1);
        } catch (IOException ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * remove a certain attribute from the dataset
     * @param id 
     */
    public static void removeAttribute(int id) {
        data.deleteAttributeAt(id);
    }
    
    /**
     * resample the instances in dataset
     * @param seed
     * @return 
     */
    public static Instances resample(int seed) {
        return data.resample(new Random(seed));
    }

    /**
     * apply all filter to build the classifier
     * @param train
     * @param Classifier
     */
    public static void buildModel(String Classifier, Instances train) {
        try {
            // Membangun model dan melakukan test
            switch (Classifier.toLowerCase()) {
                case "naivebayes" :
                    classifier = (Classifier) new NaiveBayes();
                    break;
                case "j48" :
                    classifier = (Classifier) new MyJ48();
                    break;
                case "id3" :
                    classifier = (Classifier) new MyID3();
                default :
                    break;
            }
            classifier.buildClassifier(train);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * save classifier to a specific location
     * @param filePath 
     */
    public static void saveClassifier(String filePath) {
        try {
            SerializationHelper.write(filePath, classifier);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * load classifier from .model file
     * @param modelPath
     */
    public static void loadClassifer(String modelPath) {
        try {
            classifier = (Classifier) SerializationHelper.read(modelPath);
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * show learning statistic result by folds-cross-validation
     * @param data instances
     * @param folds num of folds
     */
    public static void FoldSchema(Instances data, int folds) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, folds, new Random(1));
            System.out.println(eval.toSummaryString("\nResults " + folds + " folds cross-validation\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * show learning statistic result by full-training method
     * @param data 
     */
    public static void FullSchema(Instances data) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(classifier, data);
            System.out.println(eval.toSummaryString("\nResults Full-Training\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * show learning statistic result by using test sets
     * @param testPath
     * @param typeTestFile
     */
    public static void TestSchema(String testPath, String typeTestFile) {
        Instances testsets = null;
        // Load test instances based on file type and path
        if (typeTestFile.equals("arff")) {
            FileReader file = null;
            try {
                file = new FileReader(testPath);
                try (BufferedReader reader = new BufferedReader(file)) {
                    testsets = new Instances(reader);
                }
                // setting class attribute
                testsets.setClassIndex(data.numAttributes() - 1);
            } catch (FileNotFoundException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                try {
                    if (file!=null) {
                        file.close();
                    }
                } catch (IOException ex) {
                    Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        } else if (typeTestFile.equals("csv")) {
            try {
                CSVLoader csv = new CSVLoader();
                csv.setFile(new File(testPath));
                data = csv.getDataSet();

                // setting class attribute
                data.setClassIndex(data.numAttributes() - 1);
            } catch (IOException ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        // Start evaluate model using instances test and print results
        try {
            Evaluation eval = new Evaluation(Util.getData());
            eval.evaluateModel(Util.getClassifier(), testsets);
            System.out.println(eval.toSummaryString("\nResults\n\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * show learning statistic result by percentage split
     * @param data
     * @param attributeIndices
     * @param trainPercent
     * @param Classifier 
     */
    public static void PercentageSplit(Instances data, String attributeIndices, double trainPercent, String Classifier) {
        try {
            int trainSize = (int) Math.round(data.numInstances()* trainPercent / 100);
            int testSize = data.numInstances() - trainSize;
            System.out.println("trainSize " + trainSize);
            System.out.println("testSize " + testSize);
            
            data.randomize(new Random(1));
            for (int i=0; i<data.numInstances(); i++) {
                System.out.println("Harakiri : " + data.instance(i));
            }
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            switch (Classifier.toLowerCase()) {
                case "naivebayes" :
                    classifier = (Classifier) new NaiveBayes();
                    break;
                case "j48" :
                    classifier = (Classifier) new MyJ48();
                    break;
                case "id3" :
                    classifier = (Classifier) new MyID3();
                    break;
                default :
                    break;
            }
            classifier.buildClassifier(train);
            
            for (int i = 0; i < test.numInstances(); i++) {
                try {
                    double pred = classifier.classifyInstance(test.instance(i));
                    System.out.print("ID: " + test.instance(i));
                    System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                    System.out.println(", predicted: " + test.classAttribute().value((int) pred));
                } catch (Exception ex) {
                    Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
                }
            }

            // Start evaluate model using instances test and print results
            try {
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } catch (Exception e) {
                e.printStackTrace();
            }

        } catch (Exception ex) { 
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    /**
     * Classify test set using pre-build model
     * @param model
     * @param test
     */
    public static void doClassify(Classifier model, Instances test) {
        test.setClassIndex(test.numAttributes() - 1);
        for (int i = 0; i < test.numInstances(); i++) {
            try {
                double pred = model.classifyInstance(test.instance(i));
                System.out.print("ID: " + test.instance(i));
                System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
                System.out.println(", predicted: " + test.classAttribute().value((int) pred));
            } catch (Exception ex) {
                Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Util.loadARFF("D:\\Weka-3-6\\data\\weather.nominal.arff");
        Util.buildModel("id3", Util.getData());
        System.out.println(Util.getClassifier());

         Util.PercentageSplit(Util.getData(), "1-4", 66, "id3");
        // Util.FullSchema(Util.getData());
       // Util.FoldSchema(Util.getData(), 5);
        //Util.TestSchema("D:\\test.arff","arff");

       // Util.loadARFF("D:\\test.arff");
       // Util.doClassify(Util.getClassifier(), Util.getData());

//        Util.PercentageSplit(ins, "1-4", 66, "J48");
//        Util.buildModel("naivebayes", Util.getData());
//        Util.FoldSchema(Util.getData(), 10);
//        Util.FullSchema(Util.getData());

//        Util.loadCSV("C:\\Program Files\\Weka-3-7\\data\\weather.nominal.csv");
//        Instances ins = Util.filterNominalToString(Util.getData(),"1-4");
//        Util.filterClassifier("NaiveBayes", ins, "1-4");
//        Util.PercentageSplit(ins, "1-4", 66, "NaiveBayes");
//        Util.FoldSchema(ins, 10);
//        Util.FullSchema(ins);
        
//        System.out.println(Util.getData());
//        Util.saveClassifier("D:\\model1.model");
//        
//          Util.loadClassifer("D:\\model1.model");
//           Util.loadARFF("D:\\test.arff");
//        Util.doClassify(Util.getClassifier(), Util.getData());
    }
    
}
