package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.*;
import weka.core.*;

import java.util.Enumeration;

/**
 * Created by rikysamuel on 9/22/2015.
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
        return null;
    }

    @Override
    public double getMeasure(String s) {
        return 0;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        ModelSelection modSelection;

        if (bBinarySplitsNominalAttribute)
            modSelection = new BinC45ModelSelection(iMinInstances, instances);
        else
            modSelection = new C45ModelSelection(iMinInstances, instances);
        if (!bReducedErrorPruning)
            root = new C45PruneableClassifierTree(modSelection, !bUnpruned, fCF,
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

    public int getSeed() {
       return seedReducedErrorPruning;
    }

    public void setSeedReducedErrorPruning(int seedReducedErrorPruning) {
        this.seedReducedErrorPruning = seedReducedErrorPruning;
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
