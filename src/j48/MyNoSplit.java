package j48;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by rikysamuel on 9/29/2015.
 */
public class MyNoSplit extends ClassifierSplitModel {

    public MyNoSplit(Distribution dist){

        m_distribution = new Distribution(dist);
        m_numSubsets = 1;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_distribution = new Distribution(instances);
        m_numSubsets = 1;
    }

    @Override
    public String leftSide(Instances data) {
        return null;
    }

    @Override
    public String rightSide(int index, Instances data) {
        return null;
    }

    @Override
    public String sourceExpression(int index, Instances data) {
        return null;
    }

    @Override
    public double[] weights(Instance instance) {
        return null;
    }

    @Override
    public int whichSubset(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public String getRevision() {
        return null;
    }
}
