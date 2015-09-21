/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.*;


/**
 *
 * @author rikysamuel
 */
public class MyID3 extends Classifier implements TechnicalInformationHandler, Sourcable {
    private Attribute currentAttribute;
    private double classLabel;
    private MyID3[] subTree;
    private double[] classDistributionAmongInstances;

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }


    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Check if classifier can handle the data
        getCapabilities().testWithFail(instances);

        // Remove missing value instance from instances
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        // Build the classifier
        buildMyID3(instances);
    }

    public void buildMyID3(Instances instances) throws Exception {
        // Check if instance is not empty
        if (instances.numInstances() == 0) {
            currentAttribute = null;
            classLabel = Instance.missingValue();
            classDistributionAmongInstances = new double[instances.numInstances()];
        } else {
            // Compute Information Gain (IG) from each attribute

        }
    }

    @Override
    public String toSource(String s) throws Exception {
        return null;
    }


    public int toSource(int id, StringBuffer buffer) throws Exception{
        return 0;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Cilvia Sianora Putri, Steve Immanuel Harnadi, and Rikysamuel");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        result.setValue(TechnicalInformation.Field.TITLE, "MyID3 implementation");

        return result;
    }

    public double classifyInstance(Instance instance) {
        return 0.0;
    }

    public String toString() {
        return "Something";
    }

    public double computeIG(Instances data, Attribute att) {
        return 0.0;
    }

    public double computeEntropy(Instances data) {
        return 0.0;
    }

    public Instances[] splitData(Instances data, Attribute att) {
        return null;
    }

    public String getRevision() {
        return "Something";
    }

    public static void main(String[] args) {
        runClassifier(new MyID3(), args);
    }


}
