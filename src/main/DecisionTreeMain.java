package main;

import general.Util;

/**
 * Created by rikysamuel on 9/28/2015.
 */
public class DecisionTreeMain {
    public static void main(String[] args) {
        Util.loadARFF("C:\\Program Files\\Weka-3-7\\data\\weather.nominal.arff");
        Util.buildModel("j48", Util.getData());
        System.out.println(Util.getClassifier());
    }
}
