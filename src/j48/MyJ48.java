package j48;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.*;

import java.util.Enumeration;

/**
 * Created by rikysamuel on 9/22/2015.
 */
public class MyJ48 extends Classifier implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler {

    @Override
    public Enumeration enumerateMeasures() {
        return null;
    }

    @Override
    public double getMeasure(String s) {
        return 0;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }

    @Override
    public int graphType() {
        return 0;
    }

    @Override
    public String graph() throws Exception {
        return null;
    }

    @Override
    public String prefix() throws Exception {
        return null;
    }

    @Override
    public String toSource(String s) throws Exception {
        return null;
    }

    @Override
    public String toSummaryString() {
        return null;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        return null;
    }
}
