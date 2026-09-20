import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# Mapping month names to integer indices (0 through 11)
MONTH_MAP = {
    "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
    "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file filename and convert into a list of
    evidence lists and a list of labels.
    """
    evidence = []
    labels = []

    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                MONTH_MAP[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbors model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)
    return classifier


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    positives_correct = 0
    negatives_correct = 0
    actual_positives = 0
    actual_negatives = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            actual_positives += 1
            if predicted == 1:
                positives_correct += 1
        else:
            actual_negatives += 1
            if predicted == 0:
                negatives_correct += 1

    sensitivity = positives_correct / actual_positives if actual_positives > 0 else 0.0
    specificity = negatives_correct / actual_negatives if actual_negatives > 0 else 0.0

    return (sensitivity, specificity)


if __name__ == "_main_":
    main()
