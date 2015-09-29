package j48;

import weka.classifiers.trees.j48.Distribution;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

import java.io.Serializable;

/**
 * Created by rikysamuel on 9/29/2015.
 */
public abstract class MyClassifierSplitModel implements Cloneable, Serializable, RevisionHandler {
    public Distribution dDistribution;
    public int iNumSubsets;
    private static final long serialVersionUID = 4280730118393457357L;

    public abstract void buildClassifier(Instances instances) throws Exception;

    public final boolean checkModel() {
         return (iNumSubsets > 0);
    }

    public double classProb(int classIndex, Instance instance, int theSubset) throws Exception {
        if (theSubset > -1) {
            return dDistribution.prob(classIndex,theSubset);
        } else {
            double [] weights = weights(instance);
            if (weights == null) {
                return dDistribution.prob(classIndex);
            } else {
                double prob = 0;
                for (int i = 0; i < weights.length; i++) {
                    prob += weights[i] * dDistribution.prob(classIndex, i);
                }
                return prob;
            }
        }
    }

    public abstract String leftSide(Instances data);

    public abstract String rightSide(int index,Instances data);

    public final String dumpLabel(int index,Instances data) throws Exception {
        StringBuffer text;

        text = new StringBuffer();
        text.append(data.classAttribute().value(dDistribution.maxClass(index)));
        text.append(" (").append(Utils.roundDouble(dDistribution.perBag(index), 2));
        if (Utils.gr(dDistribution.numIncorrect(index),0)) {
            text.append("/").append(Utils.roundDouble(dDistribution.numIncorrect(index), 2));
        }
        text.append(")");

        return text.toString();
    }

    public final Instances [] split(Instances data) throws Exception {
        Instances [] instances = new Instances [iNumSubsets];
        double [] weights;
        double newWeight;
        Instance instance;
        int subset, i, j;

        for (j=0;j<iNumSubsets;j++) {
            instances[j] = new Instances(data, data.numInstances());
        }

        for (i = 0; i < data.numInstances(); i++) {
            instance = data.instance(i);
            weights = weights(instance);
            subset = whichSubset(instance);
            if (subset > -1) {
                instances[subset].add(instance);
            }
            else {
                for (j = 0; j < iNumSubsets; j++) {
                    if (Utils.gr(weights[j],0)) {
                        newWeight = weights[j]*instance.weight();
                        instances[j].add(instance);
                        instances[j].lastInstance().setWeight(newWeight);
                    }
                }
            }
        }

        for (j = 0; j < iNumSubsets; j++)
            instances[j].compactify();

        return instances;
    }

    public abstract double [] weights(Instance instance);

    public abstract int whichSubset(Instance instance) throws Exception;

    @Override
    public String getRevision() {
        return null;
    }
}
