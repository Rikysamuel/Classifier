package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.Distribution;
import weka.core.*;

import java.util.Enumeration;

/**
 * Created by Rikysamuel on 9/22/2015.
  ^ sombong
 */
public class MyJ48 extends Classifier implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler {

    private int testI;
    private boolean bEmpty;
    private boolean bLeaf;
    private boolean bPruneTree = true;
    private MyClassifierSplitModel csmLocalModel;
    private MyJ48[] ctSons;
    private Instances dataInstances;
    private float dConfFact = 0.25f;

    public MyJ48(boolean bPruneTree) {
        this.bPruneTree = bPruneTree;
    }

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

    public final MyClassifierSplitModel selectModel(Instances data) {
        dataInstances = new Instances(data);
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
                        return noSplitModel;
                    } else {
                        if (bestModel!= null) {
                            bestModel.dDistribution.addInstWithUnknown(data, bestModel.iAttIndex);
                            bestModel.setSplitPoint(data);
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
        bLeaf = false;
        bEmpty = false;
        ctSons = null;
        buildMyJ48(instances);

        collapse();
        if(bPruneTree) {
            prune();
        }
    }

    public void buildMyJ48(Instances instances) throws Exception {
        csmLocalModel = selectModel(instances);
        if(csmLocalModel.iNumSubsets > 1) {
            Instances[] localInstances = csmLocalModel.split(instances);
            ctSons = new MyJ48[csmLocalModel.iNumSubsets];

            for(int i = 0; i < ctSons.length; ++i) {
                ctSons[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        } else {
            bLeaf = true;
            if(Utils.eq(instances.sumOfWeights(), 0.0D)) {
                bEmpty = true;
            }
        }
    }

    public MyJ48 getNewTree(Instances data) throws Exception {
        MyJ48 newTree = new MyJ48(bPruneTree);
        newTree.buildMyJ48(data);

        return newTree;
    }

    public void prune() {
        double dErrLeaf;
        double dErrTree;

        if (!bLeaf){
            for (int i=0;i<ctSons.length;i++)
                ctSons[i].prune();

            // count error as leaf and tree
            dErrLeaf = getErrorInLeaf(csmLocalModel.dDistribution);
            dErrTree = getErrorInTree();

            if (dErrLeaf<=dErrTree+0.1){
                ctSons = null;
                bLeaf = true;
                csmLocalModel = new MyNoSplit(csmLocalModel.dDistribution);
                return;
            }
        }
    }

    public double getErrorInTree(){
        double err = 0;
        if (bLeaf)
            return getErrorInLeaf(csmLocalModel.dDistribution);
        else{
            for (int i=0;i<ctSons.length;i++) {
                err = err + ctSons[i].getErrorInTree();
            }
            return err;
        }
    }

    public double getErrorInLeaf(Distribution dist){
        if (dist.total()==0)
            return 0;
        else
            return dist.numIncorrect()+addErrs(dist.total(),dist.numIncorrect(),dConfFact);
    }

    public double addErrs(double N, double e, float CF){

        // Ignore stupid values for CF
        if (CF > 0.5) {
            System.err.println("WARNING: confidence value for pruning " +
                    " too high. Error estimate not modified.");
            return 0;
        }

        // Check for extreme cases at the low end because the
        // normal approximation won't work
        if (e < 1) {

            // Base case (i.e. e == 0) from documenta Geigy Scientific
            // Tables, 6th edition, page 185
            double base = N * (1 - Math.pow(CF, 1 / N));
            if (e == 0) {
                return base;
            }

            // Use linear interpolation between 0 and 1 like C4.5 does
            return base + e * (addErrs(N, 1, CF) - base);
        }

        // Use linear interpolation at the high end (i.e. between N - 0.5
        // and N) because of the continuity correction
        if (e + 0.5 >= N) {

            // Make sure that we never return anything smaller than zero
            return Math.max(N - e, 0);
        }

        // Get z-score corresponding to CF
        double z = Statistics.normalInverse(1 - CF);

        // Compute upper limit of confidence interval
        double  f = (e + 0.5) / N;
        double r = (f + (z * z) / (2 * N) +
                z * Math.sqrt((f / N) -
                        (f * f / N) +
                        (z * z / (4 * N * N)))) /
                (1 + (z * z) / N);

        return (r * N) - e;
    }

    public void collapse() {
        if(!bLeaf) {
            double errorsOfSubtree = getTrainingErrors();
            double errorsOfTree = csmLocalModel.dDistribution.numIncorrect();
            if(errorsOfSubtree >= errorsOfTree - 0.001D) {
                ctSons = null;
                bLeaf = true;
                csmLocalModel = new MyNoSplit(csmLocalModel.dDistribution);
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
            return csmLocalModel.dDistribution.numIncorrect();
        else{
            for (i=0;i<ctSons.length;i++)
                errors = errors + ctSons[i].getTrainingErrors();
            return errors;
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double maxProb = -1;
        double currentProb;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProb = getProbs(j, instance, 1);
            if (Utils.gr(currentProb,maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }

        return (double)maxIndex;
    }

    private double getProbs(int classIndex, Instance instance, double weight) throws Exception {
        double prob = 0;

        if (bLeaf) {
            return weight * csmLocalModel.classProb(classIndex, instance, -1);
        } else {
            int treeIndex = csmLocalModel.whichSubset(instance);
            if (treeIndex == -1) {
                double[] weights = csmLocalModel.weights(instance);
                for (int i = 0; i < ctSons.length; i++) {
                    if (!ctSons[i].bEmpty) {
                        prob += ctSons[i].getProbs(classIndex, instance, weights[i] * weight);
                    }
                }
                return prob;
            } else {
                if (ctSons[treeIndex].bEmpty) {
                    return weight * csmLocalModel.classProb(classIndex, instance,
                            treeIndex);
                } else {
                    return ctSons[treeIndex].getProbs(classIndex, instance, weight);
                }
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double [] doubles = new double[instance.numClasses()];

        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = getProbs(i, instance, 1);
        }

        return doubles;
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
        try {
            StringBuffer text = new StringBuffer();
            text.append("J48 prune=").append(bPruneTree).append("\n\n");

            if (bLeaf) {
                text.append(": ");
                text.append(csmLocalModel.dumpLabel(0, dataInstances));
            }else {
                dumpTree(0, text);
            }
            text.append("\n\nNumber of Leaves  : \t"+numLeaves()+"\n");
            text.append("\nSize of the tree : \t"+numNodes()+"\n");

            return text.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Can't print classification tree.";
        }
    }

    private void dumpTree(int depth, StringBuffer text) throws Exception {
        for (int i=0; i<ctSons.length; i++) {
            text.append("\n");

            for (int j=0;j<depth;j++) {
                text.append("|   ");
            }

            text.append(csmLocalModel.leftSide(dataInstances));
            text.append(csmLocalModel.rightSide(i, dataInstances));

            if (ctSons[i].bLeaf) {
                text.append(": ");
                text.append(csmLocalModel.dumpLabel(i, dataInstances));
            } else {
                ctSons[i].dumpTree(depth+1, text);
            }
        }
    }

    public int numLeaves() {
        int num = 0;

        if (bLeaf) {
            return 1;
        }
        else {
            for (int i=0;i<ctSons.length;i++) {
                num = num+ctSons[i].numLeaves();
            }

            return num;
        }

    }

    public int numNodes() {
        int nodes = 1;

        if (bLeaf) {
            return 1;
        }
        else {
            for (int i=0;i<ctSons.length;i++) {
                nodes = nodes + ctSons[i].numNodes();
            }

            return nodes;
        }
    }
}
