package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.*;

import weka.core.*;

import java.util.Enumeration;
import java.util.Vector;

/**
 * Created by Rikysamuel on 9/22/2015.
  ^ sombong
 */
public class MyJ48 extends Classifier implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler {

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
    private boolean bPruneTheTree = true;

    private boolean bEmpty;
    private boolean bLeaf;
    private ModelSelection msToSelectModel;
    private ClassifierSplitModel csmLocalModel;
    private Distribution dTest;
    private MyJ48[] ctSons;

    private int minInstances = 2;
    private Instances dataInstances;

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        result.setMinimumNumberInstances(0);
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

    public final ClassifierSplitModel selectModel(Instances data) {
        MyC45Split bestModel = null;
        NoSplit noSplitModel;
        double averageInfoGain = 0.0D;
        int validModels = 0;

        try {
            Distribution checkDistribution = new Distribution(data);
            noSplitModel = new NoSplit(checkDistribution);
            // cek jika total (jum. instances) kebih besar dari minimal instances dan cek jika seluruh instance tidak masuk ke satu class saja
            if(!Utils.sm(checkDistribution.total(), (double)(2 * minInstances)) && !Utils.eq(checkDistribution.total(), checkDistribution.perClass(checkDistribution.maxClass()))) {

                MyC45Split[] currentModel = new MyC45Split[data.numAttributes()];
                double sumOfWeights = data.sumOfWeights();

                int i;
                for(i = 0; i < data.numAttributes(); ++i) {
                    if(i != data.classIndex()) {
                        currentModel[i] = new MyC45Split(i, minInstances, sumOfWeights);
                        currentModel[i].buildClassifier(data);
                        if(currentModel[i].checkModel()) {
                            averageInfoGain += currentModel[i].dInfoGain;
                            ++validModels;
                        }
                    } else {
                        currentModel[i] = null;
                    }
                }

                if(validModels == 0) {
                    return noSplitModel;
                } else {
                    averageInfoGain /= (double)validModels;
                    double minResult = 0.0D;

                    for(i = 0; i < data.numAttributes(); ++i) {
                        if(i != data.classIndex() && currentModel[i].checkModel() && currentModel[i].dInfoGain >= averageInfoGain - 0.001D && Utils.gr(currentModel[i].dGainRatio, minResult)) {
                            bestModel = currentModel[i];
                            minResult = currentModel[i].dGainRatio;
                        }
                    }

                    if(Utils.eq(minResult, 0.0D)) {
                        return noSplitModel;
                    } else {
                        if (bestModel!= null) {
                            bestModel.distribution().addInstWithUnknown(data, bestModel.iAttIndex);
                            if(dataInstances != null) {
                                bestModel.setSplitPoint(dataInstances);
                            }
                        }

                        return bestModel;
                    }
                }
            } else {
                return noSplitModel;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        instances.deleteWithMissingClass();

        // build tree
        buildMyJ48(instances);

        collapse();
        if(bPruneTheTree) {
            prune();
        }
    }

    public void buildMyJ48(Instances instances) throws Exception {
        bLeaf = false;
        bEmpty = false;
        ctSons = null;

        csmLocalModel = selectModel(instances);
        if(csmLocalModel.numSubsets() > 1) {
            Instances[] localInstances = csmLocalModel.split(instances);
            instances = null;
            ctSons = new MyJ48[csmLocalModel.numSubsets()];

            for(int i = 0; i < ctSons.length; ++i) {
                ctSons[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        } else {
            bLeaf = true;
            if(Utils.eq(instances.sumOfWeights(), 0.0D)) {
                bEmpty = true;
            }

            instances = null;
        }
    }

    public MyJ48 getNewTree(Instances data) throws Exception {
        MyJ48 newTree = new MyJ48();
        newTree.buildMyJ48(data);

        return newTree;
    }

    public void prune() {
        /*
        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        C45PruneableClassifierTree largestBranch;
        int i;

        if (!bLeaf){
            // Prune all subtrees.
            for (i=0;i<ctSons.length;i++)
                ctSons[i].prune();

            // Compute error for largest branch
            indexOfLargestBranch = csmLocalModel.distribution().maxBag();
            if (bSubtreeRaising) {
                errorsLargestBranch = ctSons[indexOfLargestBranch].getEstimatedErrorsForBranch((Instances) m_train);
            } else {
                errorsLargestBranch = Double.MAX_VALUE;
            }

            // Compute error if this Tree would be leaf
            errorsLeaf =
                    getEstimatedErrorsForDistribution(localModel().distribution());

            // Compute error for the whole subtree
            errorsTree = getEstimatedErrors();

            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
                    Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){

                // Free son Trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for node.
                m_localModel = new NoSplit(localModel().distribution());
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
                largestBranch = son(indexOfLargestBranch);
                m_sons = largestBranch.m_sons;
                m_localModel = largestBranch.localModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                prune();
            }
        }
        */
    }

    public void collapse() {
        if(!bLeaf) {
            double errorsOfSubtree = getTrainingErrors();
            double errorsOfTree = csmLocalModel.distribution().numIncorrect();
            if(errorsOfSubtree >= errorsOfTree - 0.001D) {
                ctSons = null;
                bLeaf = true;
                csmLocalModel = new NoSplit(csmLocalModel.distribution());
            } else {
                for(int i = 0; i < ctSons.length; ++i) {
                    ctSons[i].collapse();
                }
            }
        }
    }

    public double getTrainingErrors() {
        double errors = 0;
        int i;

        if (bLeaf)
            return csmLocalModel.distribution().numIncorrect();
        else{
            for (i=0;i<ctSons.length;i++)
                errors = errors + ctSons[i].getTrainingErrors();
            return errors;
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
