/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package id3;

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

        // Build the classifier
        buildMyID3(instances);
    }

    /**
     * Algoritma pohon keputusan
     * @param instances data train
     * @throws Exception
     */
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
        return 0.0;
    }

    /**
     * Menghitung entropy
     * @param data instance
     * @return hasil perhitungan entropy
     */
    public double computeEntropy(Instances data) {
        return 0.0;
    }

    /**
     * Membagi dataset menurut atribut value
     * @param data instance
     * @param att atribut input
     * @return instance hasil split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        return null;
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
        runClassifier(new MyID3(), args);
    }


}
