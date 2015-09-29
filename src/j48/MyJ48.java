package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.OptionHandler;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;
import java.util.Enumeration;
import java.util.Vector;

/**
 * Created by Rikysamuel on 9/22/2015.
 */
public class MyJ48 extends Classifier implements OptionHandler, Drawable, Matchable, Sourcable,
        WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler {

    private ClassifierTree root;
    private boolean bUnpruned = false;
    private float fCF = 0.25f;
    private int iMinInstances = 2;
    private boolean bEnableLaplace = false;
    private boolean bReducedErrorPruning = false;
    private int iReducedErrorPruningNumFolds = 3;
    private boolean bBinarySplitsNominalAttribute = false;
    private boolean bSubtreeRaising = true;
    private boolean bNoCleanup = false;
    private int seedReducedErrorPruning = 1;


    @Override
    public Capabilities getCapabilities() {
        Capabilities      result;

        try {
            if (!bReducedErrorPruning)
                result = new MyC45PruneableClassifierTree(null, !bUnpruned, fCF, bSubtreeRaising,
                        !bNoCleanup).getCapabilities();
            else
                result = new PruneableClassifierTree(null, !bUnpruned, iReducedErrorPruningNumFolds, !bNoCleanup,
                        seedReducedErrorPruning).getCapabilities();
        }
        catch (Exception e) {
            result = new Capabilities(this);
        }

        result.setOwner(this);

        return result;
    }

    @Override
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }

    @Override
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return measureNumRules();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return measureTreeSize();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return measureNumLeaves();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (j48)");
        }
    }

    public double measureNumRules() {
        return root.numLeaves();
    }

    public double measureNumLeaves() {
        return root.numLeaves();
    }

    public double measureTreeSize() {
        return root.numNodes();
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        ModelSelection modSelection;

        if (bBinarySplitsNominalAttribute)
            modSelection = new BinC45ModelSelection(iMinInstances, instances);
        else
            modSelection = new C45ModelSelection(iMinInstances, instances);
        if (!bReducedErrorPruning)
            root = new MyC45PruneableClassifierTree(modSelection, !bUnpruned, fCF,
                    bSubtreeRaising, !bNoCleanup);
        else
            root = new PruneableClassifierTree(modSelection, !bUnpruned, iReducedErrorPruningNumFolds,
                    !bNoCleanup, seedReducedErrorPruning);
        root.buildClassifier(instances);
        if (bBinarySplitsNominalAttribute) {
            ((BinC45ModelSelection)modSelection).cleanup();
        } else {
            ((C45ModelSelection)modSelection).cleanup();
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return root.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return root.distributionForInstance(instance, bEnableLaplace);
    }

    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    @Override
    public String graph() throws Exception {
        return root.graph();
    }

    @Override
    public String prefix() throws Exception {
        return root.prefix();
    }

    @Override
    public String toSource(String classAttribute) throws Exception {
        StringBuffer [] source = root.toSource(classAttribute);
        return "class ".concat(classAttribute).concat( " {\n\n")
                        .concat("  public static double classify(Object[] i)\n")
                        .concat("    throws Exception {\n\n")
                        .concat("    double p = Double.NaN;\n")+ source[0]  // Assignment code
                        + "    return p;\n"
                        .concat("  }\n") + source[1]  // Support code
                        + "}\n";
    }

    @Override
    public String toSummaryString() {
        return null;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Cilvia Sianora Putri, Steve Immanuel Harnadi, and Rikysamuel");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        result.setValue(TechnicalInformation.Field.TITLE, "MyID3 implementation");

        return result;
    }

    public boolean isbUnpruned() {
        return bUnpruned;
    }

    public void setbUnpruned(boolean bUnpruned) {
        this.bUnpruned = bUnpruned;
    }

    public float getfCF() {
        return fCF;
    }

    public void setfCF(float fCF) {
        this.fCF = fCF;
    }

    public int getiMinInstances() {
        return iMinInstances;
    }

    public void setiMinInstances(int iMinInstances) {
        this.iMinInstances = iMinInstances;
    }

    public boolean isbReducedErrorPruning() {
        return bReducedErrorPruning;
    }

    public void setbReducedErrorPruning(boolean bReducedErrorPruning) {
        if (bReducedErrorPruning) {
            bUnpruned = false;
        }
        this.bReducedErrorPruning = bReducedErrorPruning;
    }

    public int getiReducedErrorPruningNumFolds() {
        return iReducedErrorPruningNumFolds;
    }

    public void setiReducedErrorPruningNumFolds(int iReducedErrorPruningNumFolds) {
        this.iReducedErrorPruningNumFolds = iReducedErrorPruningNumFolds;
    }

    public boolean isbBinarySplitsNominalAttribute() {
        return bBinarySplitsNominalAttribute;
    }

    public void setbBinarySplitsNominalAttribute(boolean bBinarySplitsNominalAttribute) {
        this.bBinarySplitsNominalAttribute = bBinarySplitsNominalAttribute;
    }

    public int getSeed() {
       return seedReducedErrorPruning;
    }

    public void setSeedReducedErrorPruning(int seedReducedErrorPruning) {
        this.seedReducedErrorPruning = seedReducedErrorPruning;
    }

    public boolean isbSubtreeRaising() {
        return bSubtreeRaising;
    }

    public void setbSubtreeRaising(boolean bSubtreeRaising) {
        this.bSubtreeRaising = bSubtreeRaising;
    }

    public boolean isbNoCleanup() {
        return bNoCleanup;
    }

    public void setbNoCleanup(boolean bNoCleanup) {
        this.bNoCleanup = bNoCleanup;
    }

    public int getSeedReducedErrorPruning() {
        return seedReducedErrorPruning;
    }

    public boolean isbEnableLaplace() {
        return bEnableLaplace;
    }

    public void setbEnableLaplace(boolean bEnableLaplace) {
        this.bEnableLaplace = bEnableLaplace;
    }

    public String toString() {

        if (root == null) {
            return "No classifier built";
        }
        if (bUnpruned)
            return "J48 unpruned tree\n------------------\n" + root.toString();
        else
            return "J48 pruned tree\n------------------\n" + root.toString();
    }


}
