# CircularPlotter

**CircularPlotter** is a Python-based visualization tool designed to plot datasets in a circular format with Bezier curves connecting data points.

*Datasets must have a 'class' column with all other columns being numerical*.

Iris dataset in dynamic circular coordinates with path highlighting:
![Example of Iris dataset plot and highlight](ex1.png)

## Features

    - Plots data points in a circular format based on the number of attributes.
    - Connects data points with Bezier curves.
    - Highlights the curves underneath the cursor.
    - Prints the datapoints at 0,0
    - Supports multiple classes with unique color per class.
    - Adjustable Bezier curve control points for inner and outer curves.
    - Integrated file-picker for easy dataset selection.

## Requirements

    - Python 3.x
    - Libraries:
        - numpy
        - pandas
        - matplotlib
        - sklearn
        - seaborn
        - tkinter

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
