import math

from function import exponentiated_error
from manager import Manager
from regression import find_classification, minimize_loss
from utils import (
    plot_best_fxns,
    plot_points_with_their_best_fxn,
    populate_db_with_deviation_results,
)

# Constant factor for the criterion
ACCEPTED_FACTOR = math.sqrt(2)

if __name__ == "__main__":
    best_path = "data/ideal.csv"
    train_path = "data/train.csv"

    # Create manager instances for training and best functions
    training_fxn_manager = Manager(csv_path=train_path)
    candidate_best_fxn_manager = Manager(csv_path=best_path)

    training_fxn_manager.write_to_sql(file="training", postfix=" (Training Function)")
    candidate_best_fxn_manager.write_to_sql(file="best", postfix=" (Best Function)")

    best_fxns = []
    for training_fxn in training_fxn_manager:
        best_fxn = minimize_loss(
            training_fxn=training_fxn,
            candidate_fxns=candidate_best_fxn_manager.functions,
            loss_fxn=exponentiated_error,
        )
        best_fxn.tolerance_factor = ACCEPTED_FACTOR
        best_fxns.append(best_fxn)

    plot_best_fxns(best_fxns, "training_and_best")

    test_path = "data/test.csv"
    test_fxn_manager = Manager(csv_path=test_path)
    test_fxn = test_fxn_manager.functions[0]

    points_with_best_fxn = []
    for point in test_fxn:
        best_fxn, delta_y = find_classification(point=point, best_fxns=best_fxns)
        result = {"point": point, "classification": best_fxn, "delta_y": delta_y}
        points_with_best_fxn.append(result)

    plot_points_with_their_best_fxn(points_with_best_fxn, "point_and_best")
    populate_db_with_deviation_results(points_with_best_fxn)

    print(
    """
    The following files were created:
        * mapping.db - point test results
        * best.db - all best functions in an SQLite database
        * training.db - all training functions in an SQLite database
        * point_and_best.html - the points with matching best functions and the distance between them
        * training_and_best.html - the training data as a scatter plot with the best fitting function as a curve

    Authored By - Blessing Adesiji
    """
    )
