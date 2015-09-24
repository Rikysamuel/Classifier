/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3;

import general.Util;
import sun.awt.SunHints;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.pmml.FieldMetaInfo;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;


/**
 *
 * @author rikysamuel
 */
public class MyID3 extends Classifier implements TechnicalInformationHandler, Sourcable {
    private Attribute currentAttribute;
    private double classLabel;
    private MyID3[] subTree;
    private double[] classDistributionAmongInstances;
    private Attribute classAttribute;

    /**
     * menentukan kapabilitas dari myID3
     * @return kapabilitas
     */
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


    /**
     * Membuat pohon keputusan
     * @param instances data train
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Check if classifier can handle the data
        getCapabilities().testWithFail(instances);

        // Remove missing value instance from instances
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        instances.setClassIndex(instances.numAttributes()-1);

        // Gather list of attribute in instances
        ArrayList<Attribute> remainingAttributes = new ArrayList<>();
        Enumeration enumAttributes = instances.enumerateAttributes();
        while (enumAttributes.hasMoreElements()) {
            remainingAttributes.add((Attribute) enumAttributes.nextElement());
        }
        // Start build classifier ID3
        buildMyID3(instances, remainingAttributes);
    }

    /**
     * Algoritma pohon keputusan
     * @param instances data train
     * @param attributes remaining attributes
     * @throws Exception
     */
    public void buildMyID3(Instances instances, ArrayList<Attribute> attributes) throws Exception {
        // Check if no instances have reached this node.
        if (instances.numInstances() == 0) {
            classAttribute = null;
            classLabel = Instance.missingValue();
            classDistributionAmongInstances = new double[instances.numClasses()];
            return;
        }
        // Check if all instances only contain one class label
        if (computeEntropy(instances) == 0) {
            currentAttribute = null;
            classDistributionAmongInstances = classDistribution(instances);
            // Labelling process at node
            for (int i=0; i<classDistributionAmongInstances.length; i++) {
                if (classDistributionAmongInstances[i] > 0) {
                    classLabel = i;
                }
            }
            classAttribute = instances.classAttribute();
        } else {
            // Compute infogain for each attribute
            double[] infoGainAttribute = new double[instances.numAttributes()];
            for (int i=0; i<instances.numAttributes(); i++) {
                infoGainAttribute[i] = computeIG(instances,instances.attribute(i));
            }
            // Choose attribute with maximum information gain
            int indexMaxInfoGain = 0;
            double maximumInfoGain = 0.0;
            for (int i=0; i<(infoGainAttribute.length-1); i++) {
                if (infoGainAttribute[i] > maximumInfoGain) {
                    maximumInfoGain = infoGainAttribute[i];
                    indexMaxInfoGain = i;
                }
            }
            currentAttribute = instances.attribute(indexMaxInfoGain);
            // Delete current attribute from remaining attribute
            ArrayList<Attribute> remainingAttributes = attributes;
            int indexAttributeDeleted = 0;
            for (int i=0; i<remainingAttributes.size(); i++) {
                //System.out.println("huhu " + remainingAttributes.get(i).index());
                //System.out.println("huhu2 " + currentAttribute.index());
                if (remainingAttributes.get(i).index() == currentAttribute.index()) {
                    indexAttributeDeleted = i;
                }
            }
            remainingAttributes.remove(indexAttributeDeleted);
            // Split instances based on currentAttribute (create branch new node)
            Instances[] instancesSplitBasedAttribute = splitData(instances,currentAttribute);
            subTree = new MyID3[currentAttribute.numValues()];
            for (int i=0; i<currentAttribute.numValues(); i++) {
                if (instancesSplitBasedAttribute[i].numInstances() == 0) {
                    classDistributionAmongInstances = classDistribution(instances);
                    // Labelling process at node
                    classLabel = 0.0;
                    double counterDistribution = 0.0;
                    for (int j=0; j<classDistributionAmongInstances.length; j++) {
                        if (classDistributionAmongInstances[j] > counterDistribution) {
                            classLabel = j;
                        }
                    }
                    classAttribute = instances.classAttribute();
                } else {
                    subTree[i] = new MyID3();
                    subTree[i].buildMyID3(instancesSplitBasedAttribute[i],remainingAttributes);
                }
            }
        }
    }

    /**
     * Algoritma untuk menghitung distribusi kelas
     * @param instances
     * @return distributionClass counter
     */
    public double[] classDistribution(Instances instances) {
        // Compute class distribution counter from instances
        double[] distributionClass = new double[instances.numClasses()];
        for (int i=0; i<instances.numInstances(); i++) {
            distributionClass[(int) instances.instance(i).classValue()]++;
        }
        return distributionClass;
    }

    @Override
    public String toSource(String className) throws Exception {
        int id;
        StringBuffer result = new StringBuffer();

        result.append("class ").append(className).append(" {\n");
        result.append("  private static void checkMissing(Object[] i, int index) {\n");
        result.append("    if (i[index] == null)\n");
        result.append("      throw new IllegalArgumentException(\"Null values "
                + "are not allowed!\");\n");
        result.append("  }\n\n");
        result.append("  public static double classify(Object[] i) {\n");
        id = 0;
        result.append("    return node").append(id).append("(i);\n");
        result.append("  }\n");
        toSource(id, result);
        result.append("}\n");

        return result.toString();
    }


    public int toSource(int id, StringBuffer buffer) throws Exception{
        int result;
        int i;
        int newID;
        StringBuffer[] subBuffers;

        buffer.append("\n");
        buffer.append("  protected static double node").append(id).append("(Object[] i) {\n");

        // leaf?
        if (currentAttribute == null) {
            result = id;
            if (Double.isNaN(classLabel)) {
                buffer.append("    return Double.NaN;");
            } else {
                buffer.append("    return ").append(currentAttribute).append(";");
            }
            if (currentAttribute != null) {
                buffer.append(" // ").append(currentAttribute.value((int) classLabel));
            }
            buffer.append("\n");
            buffer.append("  }\n");
        } else {
            buffer.append("    checkMissing(i, ").append(currentAttribute.index()).append(");\n\n");
            buffer.append("    // ").append(currentAttribute.name()).append("\n");

            // subtree calls
            subBuffers = new StringBuffer[currentAttribute.numValues()];
            newID = id;
            for (i = 0; i < currentAttribute.numValues(); i++) {
                newID++;

                buffer.append("    ");
                if (i > 0) {
                    buffer.append("else ");
                }
                buffer.append("if (((String) i[")
                        .append(currentAttribute.index())
                        .append("]).equals(\"")
                        .append(currentAttribute.value(i))
                        .append("\"))\n");
                buffer.append("      return node").append(newID).append("(i);\n");

                subBuffers[i] = new StringBuffer();
                newID = subTree[i].toSource(newID, subBuffers[i]);
            }
            buffer.append("    else\n");
            buffer.append("      throw new IllegalArgumentException(\"Value '\" + i[")
                    .append(currentAttribute.index()).append("] + \"' is not allowed!\");\n");
            buffer.append("  }\n");

            // output subtree code
            for (i = 0; i < currentAttribute.numValues(); i++) {
                buffer.append(subBuffers[i].toString());
            }

            result = newID;
        }

        return result;
    }

    /**
     * Informasi mengenai pembuat, tahun pembuatan, dan judul
     * @return technical information
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Cilvia Sianora Putri, Steve Immanuel Harnadi, and Rikysamuel");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        result.setValue(TechnicalInformation.Field.TITLE, "MyID3 implementation");

        return result;
    }

    /**
     * Mengklasifikasikan instance
     * @param instance data yang ingin di klasifikasikan
     * @return hasil klasifikasi
     * @throws NoSupportForMissingValuesException
     */
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Cannot handle missing value");
        }
        if (currentAttribute == null) {
            return classLabel;
        } else {
            return subTree[(int) instance.value(currentAttribute)].classifyInstance(instance);
        }
    }

    /**
     * Menghitung pendistribusian class dalam instances
     * @param instance data yang ingin dihitung distribusinya
     * @return distribusi kelas dari instance
     * @throws NoSupportForMissingValuesException
     */
    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Cannot handle missing value");
        }
        if (currentAttribute == null) {
            return classDistributionAmongInstances;
        } else {
            return subTree[(int) instance.value(currentAttribute)].
                    distributionForInstance(instance);
        }
    }

    public String toString() {
        if ((classDistributionAmongInstances == null) && (subTree== null)) {
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }

    public String toString(int level) {
        StringBuilder text = new StringBuilder();

        if (currentAttribute == null) {
            if (Instance.isMissingValue(classLabel)) {
                text.append(": null");
            } else {
                text.append(": ").append(classAttribute.value((int) classLabel));
            }
        } else {
            for (int j = 0; j < currentAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(currentAttribute.name()).append(" = ").append(currentAttribute.value(j));
                text.append(subTree[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    /**
     * Menghitung information gain
     * @param data instance
     * @param att atribut
     * @return hasil information gain
     */
    public double computeIG(Instances data, Attribute att) {
        // Split instances based on attribute values
        Instances[] instancesSplitBasedAttribute = splitData(data, att);
        // Compute information gain based on instancesSplitBasedAttribute
        double entrophyOverall = computeEntropy(data);
        for (int i=0; i<instancesSplitBasedAttribute.length; i++) {
            entrophyOverall -= ((double) instancesSplitBasedAttribute[i].numInstances() / (double) data.numInstances())
                    * computeEntropy(instancesSplitBasedAttribute[i]);
        }
        return entrophyOverall;
    }

    /**
     * Menghitung entropy
     * @param data instance
     * @return hasil perhitungan entropy
     */
    public double computeEntropy(Instances data) {
        // Compute class distribution counter from instances
        double[] distributionClass = classDistribution(data);
        // Compute entrophy from class distribution counter
        double entrophy = 0.0;
        for (int i=0; i<distributionClass.length; i++) {
            double operanLog2 = distributionClass[i] / (double) data.numInstances();
            if (operanLog2 != 0) {
                entrophy -= (distributionClass[i] / (double) data.numInstances())
                        * (Math.log(operanLog2) / Math.log(2));
            } else {
                entrophy -= 0;
            }
        }
        return entrophy;
    }

    /**
     * Membagi dataset menurut atribut value
     * @param data instance
     * @param att atribut input
     * @return instance hasil split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] instancesSplitBasedAttribute = new Instances[att.numValues()];
        for (int i=0; i<att.numValues(); i++) {
            instancesSplitBasedAttribute[i] = new Instances(data,data.numInstances());
        }
        for (int i=0; i<data.numInstances(); i++) {
            instancesSplitBasedAttribute[(int) data.instance(i).value(att)].add(data.instance(i));
        }
        return instancesSplitBasedAttribute;
    }

    /**
     * Mengembalikan informasi revision
     * @return revision info
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 0 $");
    }

    /**
     * Main method
     * @param args arguments
     */
    public static void main(String[] args) {
        // runClassifier(new MyID3(), args);
        Instances instances;
        try {
            BufferedReader reader = new BufferedReader(
                    new FileReader("D:\\Weka-3-6\\data\\weather.nominal.arff"));
            try {
                instances = new Instances(reader);
                instances.setClassIndex(instances.numAttributes() - 1);
                MyID3 id3 = new MyID3();
                try {
                    id3.buildClassifier(instances);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                // Test class distribution
                double[] classDistribution = id3.classDistribution(instances);
                for (int i=0; i<classDistribution.length; i++) {
                    System.out.println(classDistribution[i]);
                }
                // Test entrophy and information gain for each attribute
                System.out.println(id3.computeEntropy(instances));
                Enumeration attributes = instances.enumerateAttributes();
                while (attributes.hasMoreElements()) {
                    System.out.println(id3.computeIG(instances,(Attribute) attributes.nextElement()));
                }
                // Test build classifier
                try {
                    id3.buildClassifier(instances);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println(id3.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }


}
