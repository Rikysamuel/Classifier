package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
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
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import java.util.Enumeration;

/**
 * Created by Rikysamuel on 9/22/2015.
  ^ sombong
 */
public class MyJ48 extends Classifier implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler {

    private int testI;
    private boolean bEmpty;
    private boolean bLeaf;
    private ClassifierSplitModel csmLocalModel;
    private MyJ48[] ctSons;

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
        return null;
    }

    @Override
    public double getMeasure(String additionalMeasureName) {
        return 0.D;
    }

    public final ClassifierSplitModel selectModel(Instances data) {
        MyC45Split bestModel = null;
        MyNoSplit noSplitModel;
        double averageInfoGain = 0.0D;
        int validModels = 0;
        int minInstances = 2;

        try {
            Distribution checkDistribution = new Distribution(data);
            noSplitModel = new MyNoSplit(checkDistribution);

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
                            testI = i;
                            minResult = currentModel[i].dGainRatio;
                        }
                    }

                    if(Utils.eq(minResult, 0.0D)) {
                        System.out.println("no split, Model: " + testI);
                        return noSplitModel;
                    } else {
                        if (bestModel!= null) {
                            bestModel.distribution().addInstWithUnknown(data, bestModel.iAttIndex);
                            bestModel.setSplitPoint(data);
                        }
                        System.out.println("split, best-model: " + testI);
                        return bestModel;
                    }
                }
            } else {
                System.out.println("no split, " + checkDistribution.maxClass());
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
        bLeaf = false;
        bEmpty = false;
        ctSons = null;
        buildMyJ48(instances);

        collapse();
//        if(bPruneTheTree) {
//            prune();
//        }
    }

    public void buildMyJ48(Instances instances) throws Exception {
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
                csmLocalModel = new MyNoSplit(csmLocalModel.distribution());
            } else {
                for (MyJ48 ctSon : ctSons) {
                    ctSon.collapse();
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
        return 0.D;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    @Override
    public String graph() throws Exception {
        return "Something";
    }

    @Override
    public String prefix() throws Exception {
        return "Something";
    }

    @Override
    public String toSource(String classAttribute) throws Exception {
        return "Something";
    }

    @Override
    public String toSummaryString() {
        return "Something";
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Cilvia Sianora Putri, Steve Immanuel Harnadi, and Rikysamuel");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        result.setValue(TechnicalInformation.Field.TITLE, "MyID3 implementation");

        return result;
    }

    public String toString() {
        return "Print Something";
    }

}
