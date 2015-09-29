package j48;

import weka.classifiers.trees.j48.Distribution;
import weka.core.*;

/**
 * Created by rikysamuel on 9/29/2015.
 */
public abstract class MyClassifierSplitModel implements Cloneable, RevisionHandler {
    public Distribution dDistribution;
    public int iNumSubsets;

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

    public final String dumpModel(Instances data) throws Exception {
        StringBuffer text;
        int i;

        text = new StringBuffer();
        for (i=0;i<iNumSubsets;i++) {
            text.append(leftSide(data)+rightSide(i,data)+": ");
            text.append(dumpLabel(i,data)+"\n");
        }
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
