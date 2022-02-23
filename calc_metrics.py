import subprocess
import numpy
import os
import re

def get_weka_evaluation(classifier, dataset):
    print("\n\n*** Weka evalutation begins ***")
    subprocess.run(["java", "-cp", ".:../../../weka.jar", "GetStatsFor", classifier, os.path.splitext(dataset)[0]])
    print("*** Weka evaluation ends ***")


for i, d in enumerate(os.listdir("RawStatistics"), 0):
    print("%d: %s" % (i, d))

classifier = os.listdir("RawStatistics")[int(input("Select classifier: "))]

print()
for i, f in enumerate(os.listdir(os.path.join("RawStatistics", classifier))):
    print("%d: %s (%d bytes)" % (i, f, os.path.getsize(os.path.join("RawStatistics", classifier, f))))

dataset = os.listdir(os.path.join("RawStatistics", classifier))[int(input("Select classifier: "))]

csvpath = os.path.join("RawStatistics", classifier, dataset)
mat = []

with open(csvpath, "r") as file:
    for line in file:
        mat.append([float(i) for i in re.sub(r",$|\n", "", line).split(",")])

mat = numpy.array(mat)

actuals = mat[:, 0]
predictions = mat[:, 1]

true_positives = numpy.sum(numpy.logical_and(predictions == 1, actuals == 1))  # (a)
false_positives = numpy.sum(numpy.logical_and(predictions == 1, actuals == 0)) # (b)
false_negatives = numpy.sum(numpy.logical_and(predictions == 0, actuals == 1)) # (c)
true_negatives = numpy.sum(numpy.logical_and(predictions == 0, actuals == 0))  # (d)

print("true_positives: ", true_positives)
print("true_negatives: ", true_negatives)
print("false_positives: ", false_positives)
print("false_negatives: ", false_negatives)

true_postitive_rate = true_positives / (true_positives + false_negatives)
true_negative_rate = true_negatives / (false_positives + true_negatives)
false_postitive_rate = false_positives / (false_positives + true_negatives)
balanced_accuracy = (true_postitive_rate + true_negative_rate) / 2

print("TPR:", true_postitive_rate)
print("TNR:", true_negative_rate)
print("FPR:", false_postitive_rate)
print("Balanced Accuracy:", balanced_accuracy)

get_weka_evaluation(classifier, dataset)