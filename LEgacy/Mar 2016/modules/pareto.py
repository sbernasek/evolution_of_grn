__author__ = 'Sebi'


def get_pareto_front(points, goal=None, retall=False):
    """
    Returns pareto front from a list of coordinates.

    Parameters:
        points (list) - list of lists containing coordinates in the objective-space
        goal (list) - boolean list of whether each dimension in the objective-space is to be maximized
        retall (bool) - if true, return list of dominated coordinates

    Returns:
        cleared (np array) - array of pareto optimal coordinates in the objective-space
        dominated (np array) - array of non-pareto-optimal coordinates
    """

    # define max and min functions
    strategies = {True: lambda x, y: x >= y, False: lambda x, y: x <= y}

    # set strategy for each dimension in objective-space
    if goal is None:
        # if no goal is provided, assume goal is to maximize all dimensions
        comparison = {i: strategies[True] for i in range(0, len(points[0]))}
    elif len(goal) == len(points[0]):
        comparison = {i: strategies[goal[i]] for i in range(0, len(points[0]))}
    else:
        print('Error: optimization goal vector is the wrong dimension')

    # define function to check if p1 dominates p2
    dominates = lambda p1, p2: all(comparison[i](x1, x2) for i, (x1, x2) in enumerate(zip(p1, p2)))

    # classify all points as included/excluded from front
    cleared, dominated = cull(points, dominates)

    if retall is True:
        return cleared, dominated
    else:
        return cleared


def cull(points, dominates):

    """
    Culling function for pareto front, based on code from Yakym Pirozhenko at StackOverflow

    Parameters:
        points (list) - list of lists containing coordinates in the objective-space
        dominates (foo) - boolean function evaluating whether one point dominates another
    """

    dominated = []  # points that are not pareto-optimal
    cleared = []  # pareto front
    remaining = points  # comparison set

    # iterate until all points have been processed
    while len(remaining) > 0:

        # select a candidate point for inclusion in the pareto front
        candidate = remaining[0]
        new_remaining = []

        # compare candidate point with all other points in the comparison set
        for other in remaining[1:]:

            # if candidate point dominates other point, add other point to dominated, otherwise retain it for comparison
            [new_remaining, dominated][dominates(candidate, other)].append(other)

        # check if any of the points in the comparison set dominate the candidate point
        if not any(dominates(other, candidate) for other in new_remaining):

            # if no points in the comparison set dominate the candidate point, add it to the pareto front
            cleared.append(candidate)

        else:
            # if other points in the comparison set dominate the candidate point, reject it
            dominated.append(candidate)

        remaining = new_remaining

    return cleared, dominated