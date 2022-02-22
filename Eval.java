import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;
import java.io.*;
import java.nio.file.*;

public class Eval {

    public static class UCIDataset {
        
        private Instances trainInstances;
        private Instances testInstances;
        private double split = 0.5;
        private String dataDir;

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

    public static UCIDataset[] getDatasets() throws FileNotFoundException, IOException {
        File[] datasetPaths = Paths.get(".", "UCI").toFile().listFiles();
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
        }
    }

    public static void main(String[] args) {
        
        try {
            getClassifierProbabilities(new AdaBoostM1(), getDatasets());




        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }
}