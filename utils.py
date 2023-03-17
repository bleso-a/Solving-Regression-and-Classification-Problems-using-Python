"""
This module contains utility functions used for plotting
and writing data to the database.
"""

import re

from bokeh.layouts import column
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from sqlalchemy import Column, Float, MetaData, String, Table, create_engine


def plot_best_fxns(best_fxns: list, file: str):
    """
    This plots all best functions

    Arguments:

    best_fxns -- list of ideal functions
    file -- the name that will be given to the resulting .html file
    """

    best_fxns.sort(key=lambda best_fxn: best_fxn.training_fxn.name, reverse=False)
    plots = []

    for best_fxn in best_fxns:
        p = plot_graph_from_two_fxns(
            line_fxn=best_fxn,
            scatter_fxn=best_fxn.training_fxn,
            exponentiated_error=best_fxn.error,
        )
        plots.append(p)

    output_file("{}.html".format(file))
    show(column(*plots))


def plot_points_with_their_best_fxn(classified_points: list, file: str):
    """
    This plots all points that have a matched classification

    Arguments:

    classified_points -- a list of dicts with "classification" and "point"
    file_name -- the name that will be given to the resulting .html file
    """

    plots = []
    for _, item in enumerate(classified_points):
        if item["classification"]:
            plots.append(plot_classification(item["point"], item["classification"]))

    output_file("{}.html".format(file))
    show(column(*plots))


def plot_graph_from_two_fxns(scatter_fxn, line_fxn, exponentiated_error):
    """
    This plots a scatter for the training function and a line for the
    best function.

    Arguments:

    scatter_fxn -- the training function
    line_fxn -- the best function
    exponentiated_error -- the squared error will be plotted in the title
    """

    fxn_one_name = scatter_fxn.name
    fxn_one_dataframe = scatter_fxn.dataframe

    fxn_two_name = line_fxn.name
    fxn_two_dataframe = line_fxn.dataframe

    exponentiated_error = round(exponentiated_error, 2)
    title = f"Training Model {fxn_one_name} v Best Model {fxn_two_name} => \
        Total Squared Error = {exponentiated_error}"

    p = figure(title=re.sub(" +", " ", title), x_axis_label="x", y_axis_label="y")
    p.scatter(
        fxn_one_dataframe["x"],
        fxn_one_dataframe["y"],
        fill_color="red",
        legend_label="Training",
    )
    p.line(
        fxn_two_dataframe["x"], fxn_two_dataframe["y"], legend_label="Best", line_width=2
    )

    return p


def plot_classification(point, best_fxn):
    """
    This plots the classification function and a point on top.
    It also displays the tolerance

    Arguments:

    point -- a dictionary with "x" and "y" values
    best_fxn -- a classification object
    """

    if best_fxn:
        classification_fxn_dataframe = best_fxn.dataframe

        point_to_str = f"({point['x']}, {round(point['y'], 2)})"
        title = f"Point {point_to_str} with classification {best_fxn.name}"
        p = figure(title=title, x_axis_label="x", y_axis_label="y")

        # Draw the best function

        p.line(
            classification_fxn_dataframe["x"],
            classification_fxn_dataframe["y"],
            line_width=2,
            line_color="black",
            legend_label="Classification Function",
        )

        # Showing the tolerance within the graph

        criterion = best_fxn.tolerance

        classification_fxn_dataframe["upper"] = (
            classification_fxn_dataframe["y"] + criterion
        )
        classification_fxn_dataframe["lower"] = (
            classification_fxn_dataframe["y"] - criterion
        )

        source = ColumnDataSource(classification_fxn_dataframe.reset_index())

        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            source=source,
            level="underlay",
            fill_alpha=0.3,
            line_width=1,
            line_color="green",
            fill_color="green",
        )

        p.add_layout(band)

        # Draw the point

        p.scatter(
            [point["x"]],
            [round(point["y"], 4)],
            fill_color="red",
            legend_label="Test point",
            size=8,
        )

        return p


def populate_db_with_deviation_results(result):
    """
    This function populates the SQLite database with the results
    of a classification computation. More importantly, it takes
    into consideration the requirements given in the assignment

    Arguments:

    result -- a list of dictionaries describing the results of a
    classification test
    """

    engine = create_engine("sqlite:///mapping.db", echo=False)
    metadata = MetaData(engine)

    mapping = Table(
        "mapping",
        metadata,
        Column("X (Test Fxn)", Float, primary_key=False),
        Column("Y (Test Fxn)", Float),
        Column("Delta Y (Test Fxn)", Float),
        Column("Number Of Best Fxns", String(50)),
    )

    metadata.create_all()

    execute_map = []
    for item in result:
        point = item["point"]
        classification = item["classification"]
        delta_y = item["delta_y"]

        classification_name = None
        if classification:
            classification_name = classification.name.replace("y", "N")
        else:
            classification_name = "-"
            delta_y = -1

        execute_map.append(
            {
                "X (Test Fxn)": point["x"],
                "Y (Test Func)": point["y"],
                "Delta Y (Test Fxn)": delta_y,
                "Number Of Best Fxns": classification_name,
            }
        )

    i = mapping.insert()
    i.execute(execute_map)
