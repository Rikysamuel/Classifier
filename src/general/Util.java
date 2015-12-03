package general;

import java.io.*;

import id3.MyID3;
import j48.MyJ48;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;

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

    private static float confidenceFactor = 0.25f;
    private static Instances data;
    private static Classifier classifier;

    public static Instances getData() {
        return data;
    }

    public static float getConfidenceFactor() {
        return confidenceFactor;
    }

    public static void setConfidenceFactor(float confidenceFactor) {
        Util.confidenceFactor = confidenceFactor;
    }

    public static Classifier getClassifier() {
        return classifier;
    }
    
    /**
     * load dataset from ARFF format
     * @param filename file path
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
     * @param id attribute id to remove
     */
    public static void removeAttribute(int id) {
        data.deleteAttributeAt(id);
    }
    
    /**
     * resample the instances in dataset
     * @param seed random seed
     * @return resampled instances
     */
    public static Instances resample(int seed) {
        return data.resample(new Random(seed));
    }

    /**
     * apply all filter to build the classifier
     * @param train data training
     * @param Classifier model
     */
    public static void buildModel(String Classifier, Instances train) {
        try {
            // Membangun model dan melakukan test
            switch (Classifier.toLowerCase()) {
                case "naivebayes" :
                    classifier = new NaiveBayes();
                    break;
                case "j48-prune" :
                    classifier = new MyJ48(true, confidenceFactor);
                    break;
                case "j48-unprune" :
                    classifier = new MyJ48(false, confidenceFactor);
                    break;
                case "id3" :
                    classifier = new MyID3();
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
     * @param filePath  filepath name
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
     * @param modelPath model file path
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
            eval.crossValidateModel(Util.getClassifier(), data, folds, new Random(1));
            System.out.println(eval.toSummaryString("\nResults " + folds + " folds cross-validation\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(Util.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * show learning statistic result by full-training method
     * @param data training data
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
     * @param testPath test path file
     * @param typeTestFile test file
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
     * @param data training data
     * @param trainPercent percentage of the training data
     * @param Classifier model
     */
    public static void PercentageSplit(Instances data, double trainPercent, String Classifier) {
        try {
            int trainSize = (int) Math.round(data.numInstances()* trainPercent / 100);
            int testSize = data.numInstances() - trainSize;
            
            data.randomize(new Random(1));

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            switch (Classifier.toLowerCase()) {
                case "naivebayes" :
                    classifier = new NaiveBayes();
                    break;
                case "j48-prune" :
                    classifier = new MyJ48(true, 0.25f);
                    break;
                case "j48-unprune" :
                    classifier = new MyJ48(false, 0f);
                    break;
                case "id3" :
                    classifier = new MyID3();
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
     * @param model model pathfile
     * @param test test file
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

}
