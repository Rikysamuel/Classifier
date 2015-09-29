package main;

import general.Util;

/**
 * Created by rikysamuel on 9/28/2015.
 */
public class DecisionTreeMain {
    public static void main(String[] args) {
//        Util.loadARFF("G:\\weather.nominal.arff");
        Util.loadARFF("C:\\Program Files\\Weka-3-7\\data\\weather.numeric.arff");
        Util.buildModel("j48", Util.getData());
        System.out.println(Util.getClassifier());
        Util.FullSchema(Util.getData());
//        Util.FoldSchema(Util.getData(),10);
//        Util.saveClassifier("G:\\modelj48.model");
//        Util.loadClassifer("G:\\modelj48.model");

//        Util.loadARFF("G:\\test.iris.arff");
//        Util.doClassify(Util.getClassifier(),Util.getData());
    }
}
