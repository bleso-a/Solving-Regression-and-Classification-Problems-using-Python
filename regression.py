from function import BestFunction


def minimize_loss(training_fxn, candidate_fxns, loss_fxn) -> BestFunction:
    """
    This returns a BestFunction based on a training function
    and a list of ideal functions

    Arguments:

    training_fxn -- the training function
    candidate_fxns -- a list of best candidate functions
    loss_fxn -- the function used to minimize the error
    """

    smallest_error = None
    fxn_with_smallest_error = None

    for fxn in candidate_fxns:
        error = loss_fxn(training_fxn, fxn)
        if smallest_error == None or error < smallest_error:
            smallest_error = error
            fxn_with_smallest_error = fxn

    return BestFunction(
        fxn=fxn_with_smallest_error, training_fxn=training_fxn, error=smallest_error
    )


def find_classification(point, best_fxns) -> tuple:
    """
    This determines whether or not a point is within the tolerance
    of a classification and returns a tuple containing the closest
    classification (if any) and the distance.

    Arguments:

    point -- a dictionary with "x" and "y" values
    best_fxns -- a list of BestFunction instances
    """

    current_lowest_distance = None
    current_lowest_classification = None

    for best_fxn in best_fxns:
        try:
            find_y_in_classification = best_fxn.find_y_from_x(point["x"])
        except IndexError:
            raise IndexError

        # Get absolute distance
        distance = abs(find_y_in_classification - point["y"])

        if abs(distance < best_fxn.tolerance):
            # This step ensures that multiple classifications are properly
            # handled and the one with lowest distance is returned

            if (current_lowest_classification == None) or (
                distance < current_lowest_distance
            ):
                current_lowest_distance = distance
                current_lowest_classification = best_fxn

    return (current_lowest_classification, current_lowest_distance)
