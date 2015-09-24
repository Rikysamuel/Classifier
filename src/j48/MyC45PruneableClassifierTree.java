package j48;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 * Created by rikysamuel on 9/23/2015.
 */
public class MyC45PruneableClassifierTree extends ClassifierTree {

    boolean bTreePrune = false;
    float fConfidenceFactor = 0.25f;
    boolean bSubtreeRaising = true;
    boolean bCleanup = true;

    public MyC45PruneableClassifierTree(ModelSelection toSelectLocModel,
                                        boolean pruneTree ,
                                        float cf,
                                        boolean raiseTree,
                                        boolean cleanup) {
        super(toSelectLocModel);

        bTreePrune = pruneTree;
        fConfidenceFactor = cf;
        bSubtreeRaising = raiseTree;
        bCleanup = cleanup;
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

        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, bSubtreeRaising || !bCleanup);
        collapse();
        if (bTreePrune) {
            prune();
        }
        if (bCleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    public final void collapse() {

    }

    public void prune() {

    }

    protected ClassifierTree getNewTree(Instances data) throws Exception {
        MyC45PruneableClassifierTree newTree =
                new MyC45PruneableClassifierTree(m_toSelectModel, bTreePrune, fConfidenceFactor,
                        bSubtreeRaising, bCleanup);
        newTree.buildTree(data, bSubtreeRaising || !bCleanup);

        return newTree;
    }

    private double getEstimatedErrors(){
        return 0.0;
    }

    private double getEstimatedErrorsForBranch(Instances data) throws Exception {
        return 0.0;
    }

    private double getEstimatedErrorsForDistribution(Distribution theDistribution){
        return 0.0;
    }

    private double getTrainingErrors() {
        return 0.0;
    }

    private ClassifierSplitModel localModel() {
        return null;
    }

    private void newDistribution(Instances data) throws Exception {

    }

    private MyC45PruneableClassifierTree son(int index) {
        return null;
    }

    public String getRevision() {
        return RevisionUtils.extract("--------------");
    }
}
