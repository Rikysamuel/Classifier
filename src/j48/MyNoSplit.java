package j48;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by rikysamuel on 9/29/2015.
 */
public class MyNoSplit extends MyClassifierSplitModel {

    public MyNoSplit(Distribution dist){

        dDistribution = new Distribution(dist);
        iNumSubsets = 1;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        dDistribution = new Distribution(instances);
        iNumSubsets = 1;
    }

    @Override
    public String leftSide(Instances data) {
        return "";
    }

    @Override
    public String rightSide(int index, Instances data) {
        return "";
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
