import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;
import java.io.*;
import java.nio.file.*;
import java.util.HashMap;

public class GetStatsFor {

    public HashMap<String, Classifier> classifierOptions;

    public GetStatsFor() {
        this.classifierOptions = new HashMap<>();
        for (Classifier classifier : Eval.classifierArray) {
            String[] s = classifier.getClass().toGenericString().split("\\.");
            String classifierName = s[s.length - 1];
            classifierOptions.put(classifierName, classifier);
        }
    }

    public String getWekaClassifierEvaluation(String classifierName, Eval.UCIDataset dataset) throws Exception {
        return getWekaClassifierEvaluation(this.classifierOptions.get(classifierName), dataset);
    }

    public String getWekaClassifierEvaluation(Classifier classifier, Eval.UCIDataset dataset) throws Exception {
        System.out.println("Using classifier: " + classifier.getClass().getSimpleName() + " Dataset: " + dataset.dataDir);

        Instances trainer = dataset.getTrainInstances();
        Instances tester = dataset.getTestInstances();
        classifier.buildClassifier(trainer);
        
        Evaluation evaluation = new Evaluation(trainer);
        evaluation.evaluateModel(classifier, tester);
        
        String out = "";
        out += "True Postitives:\t\t" + evaluation.numTruePositives(0) + "\n";
        out += "True Negatives:\t\t" + evaluation.numTrueNegatives(0) + "\n";
        out += "False Positives:\t\t" + evaluation.numFalsePositives(0) + "\n";
        out += "False Negatives:\t\t" + evaluation.numFalseNegatives(0) + "\n";
        return out;
        // return evaluation.toSummaryString() + "True Positive Rate\t\t\t" + evaluation.truePositiveRate(0) + "\nFalse Positive Rate\t\t\t" + evaluation.falsePositiveRate(0);
    }
    
    public static void main(String[] args) {
        GetStatsFor gsf = new GetStatsFor();
        // System.out.println(args[0]);
        // System.out.println(args[1]);

        // System.out.println("Key options:");
        // for (String key : gsf.classifierOptions.keySet()) {
        //     System.out.println(key);
        // }
        try {
            System.out.println(gsf.getWekaClassifierEvaluation(
                args[0], 
                new Eval.UCIDataset(Paths.get(".", "UCI", args[1]).toString()))
            );
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
