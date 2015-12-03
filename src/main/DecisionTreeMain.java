package main;

import general.Util;

/**
 * Created by rikysamuel on 9/28/2015.
 */
public class DecisionTreeMain {
    public static void main(String[] args) {
        Util.loadARFF("D:\\Weka-3-6\\data\\weather.numeric.arff");
        Util.setConfidenceFactor(0.25f);
     //   Util.buildModel("j48-unprune", Util.getData());
        Util.buildModel("j48-prune", Util.getData());
     //   Util.buildModel("id3", Util.getData());

     //   Util.FoldSchema(Util.getData(),10);
     //   Util.FullSchema(Util.getData());
//        Util.TestSchema("G:\\testmodel.iris.arff","arff");
        Util.PercentageSplit(Util.getData(),66,"j48-prune");

//        Util.FoldSchema(Util.getData(),5);
//        System.out.println(Util.getClassifier());
//        Util.FullSchema(Util.getData());
//        Util.FoldSchema(Util.getData(),5);
//        Util.TestSchema("G:\\testmodel.iris.arff","arff");
      //  Util.PercentageSplit(Util.getData(),"1-4",66, "j48-unprune");
    }
}
