import weka.core.Instances;
import java.io.*;

public class Utilities {

    public static Instances loadData(String arffPath) throws FileNotFoundException, IOException {
        return new Instances(new FileReader(arffPath));
    }

    public static Instances[] splitInstances(Instances all, double split) {
        // might be prudent to shuffle first- possibly can lead to biases
        int splitIndex = (int)(all.numInstances()  * split);
        Instances trainer = new Instances(all, 0);
        Instances tester = new Instances(all, 0);

        // there's probably a way to do this in O(1) not O(n)
        for (int i = 0; i < all.numInstances(); i++) {
            if (i < splitIndex) {
                trainer.add(all.instance(i));
            } else {
                tester.add(all.instance(i));
            }
        };

        return new Instances[]{trainer, tester};
    }

    public static void main(String[] args) {
        try {
            Instances both = loadData("JW_RedVsBlack.arff");
            splitInstances(both, 0.5);


        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }
}
