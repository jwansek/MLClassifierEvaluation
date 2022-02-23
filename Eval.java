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

public class Eval {

    public static Classifier[] classifierArray = new Classifier[]{new J48(), new AdaBoostM1(), new BayesNet(), new DecisionStump(), new HoeffdingTree(), new IBk(), new LMT(), new LogitBoost(), new NaiveBayes(), new NaiveBayesMultinomialText(), new RandomForest(), new RandomTree(), new REPTree()};

    public static class UCIDataset {
        
        private Instances trainInstances;
        private Instances testInstances;
        private double split = 0.5;
        public String dataDir;

        public UCIDataset(String dataDir, double split) throws FileNotFoundException, IOException {
            this.dataDir = dataDir;
            this.split = split;
            init();
        }

        public UCIDataset(String dataDir) throws FileNotFoundException, IOException {
            this.dataDir = dataDir;
            init();
        }

        private void init() throws FileNotFoundException, IOException {
            Instances bothInstances = Utilities.loadData(Paths.get(dataDir, getName() + ".arff").toString());
            // System.out.println(getName() + " " + (int)(bothInstances.numInstances()  * 0.5));
            Instances[] splitted = Utilities.splitInstances(bothInstances, split);

            this.trainInstances = splitted[0];
            this.testInstances = splitted[1];

            this.trainInstances.setClassIndex(this.trainInstances.numAttributes() - 1);
            this.testInstances.setClassIndex(this.testInstances.numAttributes() - 1);
        }

        public Instances getTrainInstances() {
            return this.trainInstances;
        }

        public Instances getTestInstances() {
            return this.testInstances;
        }

        public String getName() {
            return Paths.get(dataDir).getName(Paths.get(dataDir).getNameCount() - 1).toString();
        }
    }

    public static UCIDataset[] getDatasets(File datasetPath) throws FileNotFoundException, IOException {
        File[] datasetPaths = datasetPath.listFiles();
        UCIDataset[] datasets = new UCIDataset[datasetPaths.length];
        for (int i = 0; i < datasetPaths.length; i++) {
            datasets[i] = new UCIDataset(datasetPaths[i].toString());
        }
        return datasets;
    }
    
    public static <T extends Classifier> void getClassifierProbabilities(T classifier, UCIDataset[] datasets) throws Exception {
        String[] s = classifier.getClass().toGenericString().split("\\.");
        String classifierName = s[s.length - 1];
        Paths.get(".", "RawStatistics", classifierName).toFile().mkdir();
        for (UCIDataset dataset : datasets) {

            Instances trainer = dataset.getTrainInstances();
            Instances tester = dataset.getTestInstances();
            classifier.buildClassifier(trainer);

            FileWriter writer = new FileWriter(Paths.get(".", "RawStatistics", classifierName, dataset.getName() + ".csv").toFile());
            
            int numCorrect = 0;
            for (Instance instance : tester) {
                int prediction = (int)classifier.classifyInstance(instance);
                int actual = (int)instance.classValue();

                if (prediction == actual) {
                    numCorrect++;
                }

                writer.write(actual + "," + prediction + ",");
                for (double probability : classifier.distributionForInstance(instance)) {
                    writer.write(probability + ",");
                }
                writer.write("\n");
            }
            System.out.println(Paths.get(".", classifierName, dataset.getName() + ".csv").toString());
            System.out.println(numCorrect + " / " + tester.numInstances() + " correct");
            writer.close();
        }
    }

    public static <T extends Classifier> void getClassifierMetrics(T[] classifiers, UCIDataset dataset) throws Exception {
        FileWriter writer = new FileWriter(Paths.get(".", "MetricsByDataset", dataset.getName() + ".csv").toFile());
            writer.write("Classifier Name,Correct,Incorrect,Kappa,Total Cost,Average Cost,Mean Absolute Error");
            writer.write("Root Mean Squared Error,Coverate,True Positive Rate,False Positive Rate,Precision,Recall,F Measure");
            writer.write("Matthew's Correlation Coefficient,Area under ROC,Area under PRC");
            writer.write("\n");

        for (T classifier : classifiers) {
            String[] s = classifier.getClass().toGenericString().split("\\.");
            String classifierName = s[s.length - 1];

            Instances trainer = dataset.getTrainInstances();
            Instances tester = dataset.getTestInstances();
            classifier.buildClassifier(trainer);
            Evaluation evaluation = new Evaluation(trainer);
            evaluation.evaluateModel(classifier, tester);

            // writer.write(classifierName + ",");
            // writer.write(evaluation.correct() + ",");
            // writer.write(evaluation.incorrect() + ",");
            // writer.write(evaluation.kappa() + ",");
            // writer.write(evaluation.totalCost() + ",");
            // writer.write(evaluation.avgCost() + ",");
            // writer.write(evaluation.meanAbsoluteError() + ",");
            // writer.write(evaluation.rootMeanSquaredError() + ",");
            // writer.write(evaluation.relativeAbsoluteError() + ",");
            // writer.write(evaluation.rootRelativeSquaredError() + ",");
            // writer.write(evaluation.coverageOfTestCasesByPredictedRegions() + ",");
            // writer.write(evaluation.truePositiveRate(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.falsePositiveRate(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.precision(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.recall(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.fMeasure(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.matthewsCorrelationCoefficient(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.areaUnderROC(tester.numAttributes() - 1) + ",");
            // writer.write(evaluation.areaUnderPRC(tester.numAttributes() - 1) + ",");

            System.out.println(classifierName);
            System.out.println(evaluation.toSummaryString());

        }
        writer.close();
    }

    public static void main(String[] args) {
        
        try {
            // getClassifierProbabilities(new AdaBoostM1(), getDatasets(Paths.get(".", "UCI").toFile()));

            getClassifierMetrics(classifierArray, new UCIDataset(Paths.get(".", "UCIContinuous", "bank").toString()));



        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }
}