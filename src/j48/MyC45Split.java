package j48;

import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.GainRatioSplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * Created by rikysamuel on 9/28/2015.
 */
public class MyC45Split extends MyClassifierSplitModel {
    private static final long serialVersionUID = 3064079330067913191L;
    public int iComplexityIndex;
    public int iAttIndex;
    public int iMinInstances;
    public double dSplitValue;
    public double dInfoGain;
    public double dGainRatio;
    public double dTotalWeights;
    public int iIndex;
    public static InfoGainSplitCrit infoGainCrit = new InfoGainSplitCrit();
    public static GainRatioSplitCrit gainRatioCrit = new GainRatioSplitCrit();

    public MyC45Split(int attIndex, int minNoObj, double sumOfWeights) {
        iAttIndex = attIndex;
        iMinInstances = minNoObj;
        dTotalWeights = sumOfWeights;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        iNumSubsets = 0;
        dSplitValue = Double.MAX_VALUE;
        dInfoGain = 0;
        dGainRatio = 0;

        // handle nominal value
        if (instances.attribute(iAttIndex).isNominal()) {
            iComplexityIndex = instances.attribute(iAttIndex).numValues();
            iIndex = iComplexityIndex;
            handleNominalAttribute(instances);
        }else{ // handle numeric value
            iComplexityIndex = 2;
            iIndex = 0;
            instances.sort(instances.attribute(iAttIndex));
            handleNumericAttribute(instances);
        }
    }

    public void handleNominalAttribute(Instances data) throws Exception {
        Instance instance;
        dDistribution = new Distribution(iComplexityIndex, data.numClasses());

        Enumeration enu = data.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (!instance.isMissing(iAttIndex)) {
                dDistribution.add((int)instance.value(iAttIndex),instance);
            }
        }

        if (dDistribution.check(iMinInstances)) {
            iNumSubsets = iComplexityIndex;
            dInfoGain = infoGainCrit.splitCritValue(dDistribution, dTotalWeights);
            dGainRatio = gainRatioCrit.splitCritValue(dDistribution,dTotalWeights, dInfoGain);
        }
    }

    public void handleNumericAttribute(Instances data) throws Exception {
        int firstMiss, i, next = 1, last = 0, splitIndex = -1;
        double currentInfoGain, defaultEnt, minSplit;
        Instance instance;

        dDistribution = new Distribution(2,data.numClasses());

        Enumeration enu = data.enumerateInstances();
        i = 0;
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (instance.isMissing(iAttIndex)) {
                break;
            }
            dDistribution.add(1,instance);
            i++;
        }
        firstMiss = i;

        minSplit =  0.1 * dDistribution.total() / ((double) data.numClasses());
        if (Utils.smOrEq(minSplit, iMinInstances)) {
            minSplit = iMinInstances;
        }
        else {
            if (Utils.gr(minSplit,25)) {
                minSplit = 25;
            }
        }

        // cek apakah jumlah instance lebih besar dari jumlah instance minimal
        if (Utils.sm((double) firstMiss, 2 * minSplit)) {
            return;
        }

        // Compute values of criteria for all possible split indices.
        defaultEnt = infoGainCrit.oldEnt(dDistribution);
        while (next < firstMiss) {

            if (data.instance(next - 1).value(iAttIndex) + 1e-5 < data.instance(next).value(iAttIndex)) {

                // Move class values for all Instances up to next possible split point.
                dDistribution.shiftRange(1,0,data,last,next);

                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(dDistribution.perBag(0), minSplit) && Utils.grOrEq(dDistribution.perBag(1), minSplit)) {
                    currentInfoGain = infoGainCrit.splitCritValue(dDistribution, dTotalWeights, defaultEnt);
                    if (Utils.gr(currentInfoGain, dInfoGain)) {
                        dInfoGain = currentInfoGain;
                        splitIndex = next - 1;
                    }
                    iIndex++;
                }
                last = next;
            }
            next++;
        }

        // Was there any useful split?
        if (iIndex == 0) { // ????????????????????????????????????????????????????????????????????????????????????????
            return;
        }

        // Compute modified information gain for best split.
        dInfoGain = dInfoGain - (Utils.log2(iIndex) / dTotalWeights);
        if (Utils.smOrEq(dInfoGain, 0)) {
            return;
        }

        // Set instance variables' values to values for best split.
        iNumSubsets = 2;
        dSplitValue = (data.instance(splitIndex+1).value(iAttIndex) + data.instance(splitIndex).value(iAttIndex)) / 2;

        // In case we have a numerical precision problem we need to choose the smaller value
        if (dSplitValue == data.instance(splitIndex + 1).value(iAttIndex)) {
            dSplitValue = data.instance(splitIndex).value(iAttIndex);
        }

        // Restore distribution for best split.
        dDistribution = new Distribution(2, data.numClasses());
        dDistribution.addRange(0, data, 0, splitIndex+1);
        dDistribution.addRange(1, data, splitIndex+1, firstMiss);

        // Compute modified gain ratio for best split.
        dGainRatio = gainRatioCrit.splitCritValue(dDistribution, dTotalWeights, dInfoGain);
    }

    public void setSplitPoint(Instances data) {
        double newSplitPoint = -Double.MAX_VALUE;
        double tempValue;
        Instance instance;

        if ((data.attribute(iAttIndex).isNumeric()) && (iNumSubsets > 1)) {
            Enumeration enu = data.enumerateInstances();
            while (enu.hasMoreElements()) {
                instance = (Instance) enu.nextElement();
                if (!instance.isMissing(iAttIndex)) {
                    tempValue = instance.value(iAttIndex);
                    if (Utils.gr(tempValue,newSplitPoint) && Utils.smOrEq(tempValue, dSplitValue)) {
                        newSplitPoint = tempValue;
                    }
                }
            }
            dSplitValue = newSplitPoint;
        }
    }

    @Override
    public String leftSide(Instances data) {
        System.out.println("iAttIndex: " + iAttIndex + " : " + data.attribute(iAttIndex).name());
        return data.attribute(iAttIndex).name();
    }

    @Override
    public String rightSide(int index, Instances data) {
        StringBuffer text;

        text = new StringBuffer();
        if (data.attribute(iAttIndex).isNominal())
            text.append(" = "+
                    data.attribute(iAttIndex).value(index));
        else
        if (index == 0)
            text.append(" <= "+
                    Utils.doubleToString(dSplitValue,6));
        else
            text.append(" > "+
                    Utils.doubleToString(dSplitValue,6));
        return text.toString();
    }

    @Override
    public double[] weights(Instance instance) {
        double [] weights;
        int i;

        if (instance.isMissing(iAttIndex)) {
            weights = new double [iNumSubsets];
            for (i=0;i<iNumSubsets;i++)
                weights [i] = dDistribution.perBag(i)/dDistribution.total();
            return weights;
        }else{
            return null;
        }
    }

    @Override
    public int whichSubset(Instance instance) throws Exception {
        if (instance.isMissing(iAttIndex))
            return -1;
        else{
            if (instance.attribute(iAttIndex).isNominal())
                return (int)instance.value(iAttIndex);
            else
            if (Utils.smOrEq(instance.value(iAttIndex), dSplitValue))
                return 0;
            else
                return 1;
        }
    }

    @Override
    public String getRevision() {
        return null;
    }
}
