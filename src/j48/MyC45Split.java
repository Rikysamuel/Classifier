package j48;

import weka.classifiers.trees.j48.ClassifierSplitModel;
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
public class MyC45Split extends ClassifierSplitModel {


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

    public MyC45Split(int attIndex,int minNoObj, double sumOfWeights) {
        iAttIndex = attIndex;
        iMinInstances = minNoObj;
        dTotalWeights = sumOfWeights;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        m_numSubsets = 0;
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
        m_distribution = new Distribution(iComplexityIndex, data.numClasses());

        Enumeration enu = data.enumerateInstances();
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (!instance.isMissing(iAttIndex)) {
                m_distribution.add((int)instance.value(iAttIndex),instance);
            }
        }

        if (m_distribution.check(iMinInstances)) {
            m_numSubsets = iComplexityIndex;
            dInfoGain = infoGainCrit.splitCritValue(m_distribution, dTotalWeights);
            dGainRatio = gainRatioCrit.splitCritValue(m_distribution,dTotalWeights, dInfoGain);
        }
    }

    public void handleNumericAttribute(Instances data) throws Exception {
        int firstMiss, i, next = 1, last = 0, splitIndex = -1;
        double currentInfoGain, defaultEnt, minSplit;
        Instance instance;

        m_distribution = new Distribution(2,data.numClasses());

        Enumeration enu = data.enumerateInstances();
        i = 0;
        while (enu.hasMoreElements()) {
            instance = (Instance) enu.nextElement();
            if (instance.isMissing(iAttIndex)) {
                break;
            }
            m_distribution.add(1,instance);
            i++;
        }
        firstMiss = i;

        /**
         *
         */
        minSplit =  0.1 * m_distribution.total() / ((double) data.numClasses()); // dari mana 0.1 ?????????????????????
        if (Utils.smOrEq(minSplit, iMinInstances)) {
            minSplit = iMinInstances;
        }
        else {
            if (Utils.gr(minSplit,25)) {
                minSplit = 25; // dari mana 25 ????????????????????????????????????????????????????????????????????????
            }
        }
        /**
         *
         */

        // cek apakah jumlah instance lebih besar dari jumlah instance minimal
        if (Utils.sm((double) firstMiss, 2 * minSplit)) {
            return;
        }

        // Compute values of criteria for all possible split indices.
        defaultEnt = infoGainCrit.oldEnt(m_distribution);
        while (next < firstMiss) {

            if (data.instance(next - 1).value(iAttIndex) + 1e-5 < data.instance(next).value(iAttIndex)) {

                // Move class values for all Instances up to next possible split point.
                m_distribution.shiftRange(1,0,data,last,next);

                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(m_distribution.perBag(0), minSplit) && Utils.grOrEq(m_distribution.perBag(1), minSplit)) {
                    currentInfoGain = infoGainCrit.splitCritValue(m_distribution, dTotalWeights, defaultEnt);
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
        m_numSubsets = 2;
        dSplitValue = (data.instance(splitIndex+1).value(iAttIndex) + data.instance(splitIndex).value(iAttIndex)) / 2;

        // In case we have a numerical precision problem we need to choose the smaller value
        if (dSplitValue == data.instance(splitIndex + 1).value(iAttIndex)) {
            dSplitValue = data.instance(splitIndex).value(iAttIndex);
        }

        // Restore distribution for best split.
        m_distribution = new Distribution(2, data.numClasses());
        m_distribution.addRange(0, data, 0, splitIndex+1);
        m_distribution.addRange(1, data, splitIndex+1, firstMiss);

        // Compute modified gain ratio for best split.
        dGainRatio = gainRatioCrit.splitCritValue(m_distribution, dTotalWeights, dInfoGain);
    }

    public void setSplitPoint(Instances data) {
        double newSplitPoint = -Double.MAX_VALUE;
        double tempValue;
        Instance instance;

        if ((data.attribute(iAttIndex).isNumeric()) && (m_numSubsets > 1)) {
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
        return new double[0];
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
