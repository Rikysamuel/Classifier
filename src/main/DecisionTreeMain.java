package main;

import general.Util;

/**
 * Created by rikysamuel on 9/28/2015.
 */
public class DecisionTreeMain {
    public static void main(String[] args) {
        Util.loadARFF("C:\\Program Files\\Weka-3-7\\data\\iris.arff");
        Util.buildModel("j48", Util.getData());
        System.out.println(Util.getClassifier());
//        Util.FullSchema(Util.getData());
//        Util.PercentageSplit(Util.getData(),"1-4",66.6, "j48");
        Util.FoldSchema(Util.getData(),10);
    }
}
