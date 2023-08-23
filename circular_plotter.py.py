import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
from scipy.spatial import ConvexHull

def adjusted_bezier_curve(p0, p1, attribute_count):
    """Calculate quadratic Bezier curve points with control points adjusted based on attribute count."""
    # Control point is the midpoint between p0 and p1 but moved towards the origin
    control_point = 0.5 * (p0 + p1)
    control_point_norm = np.linalg.norm(control_point)
    
    # Adjust the control point's distance from the center based on attribute count
    adjustment_factor = 0.1 + 0.5 / attribute_count
    control_point = control_point * (1 - adjustment_factor / control_point_norm)
    
    # Calculate the Bezier curve points
    t_values = np.linspace(0, 1, 10)
    curve_points = []
    for t in t_values:
        point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control_point + t ** 2 * p1
        curve_points.append(point)
    
    return np.array(curve_points)

class SCCWithChords:
    def __init__(self, dataframe):
        self.data = dataframe
        self.positions = []
        self.colors = []
        self.transform_data()
        
    def transform_data(self):
        # The transformation code remains the same as the previous class
        attribute_count = self.data.shape[1] - 1
        scaler = MinMaxScaler((0, 1))
        
        class_column_index = self.data.columns.get_loc('class')
        numeric_data = self.data.drop(columns='class')
        class_data = self.data.iloc[:, class_column_index]
        numeric_data = scaler.fit_transform(numeric_data)
        self.data.iloc[:, [i for i in range(self.data.shape[1]) if i != class_column_index]] = numeric_data

        section_array = np.linspace(0, 1, attribute_count)
        classes = self.data['class'].unique()
        color_palette = sns.color_palette('husl', len(classes))
        self.color_map = dict(zip(classes, color_palette))

        self.all_positions = []
        self.all_colors = []

        for class_name in classes:
            positions = []
            df_name = self.data[self.data['class'] == class_name]
            df_name = df_name.drop(columns='class', axis=1)
            x_coord = np.tile(section_array, reps=len(df_name.index))
            y_coord = df_name.to_numpy().ravel()
            attribute_length = attribute_count * len(df_name)
            arc_length = 0
            arc_rule = 0
            for i in range(attribute_length):
                if arc_rule >= attribute_count:
                    arc_length = 0
                    arc_rule = 0
                arc_length += y_coord[i]
                radius = attribute_count / (2 * np.pi)
                center_angle = arc_length * 360 / (2 * np.pi * radius)
                center_angle = np.pi * center_angle / 180
                x_coord[i] = radius * np.sin(center_angle)
                y_coord[i] = radius * np.cos(center_angle)
                arc_rule += 1
                arc_length = arc_rule
            pos_array = np.column_stack((x_coord, y_coord))
            positions.extend(pos_array)
            colors = [self.color_map[class_name]] * len(pos_array)
            self.all_positions.append(positions)
            self.all_colors.append(colors)

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        attribute_count = self.data.shape[1] - 1
        
        for positions, colors in zip(self.all_positions, self.all_colors):
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], color=colors, s=20, alpha=0.5)
            
            for i in range(len(positions) - 1):
                curve_points = adjusted_bezier_curve(positions[i], positions[i+1], attribute_count)
                ax.plot(curve_points[:, 0], curve_points[:, 1], color=colors[0], alpha=0.5)
        
        circle_radius = attribute_count / (2 * np.pi)
        circle = plt.Circle((0, 0), circle_radius, color='darkgrey', fill=False)
        ax.add_artist(circle)
        
        # Adjust the label positioning and color
        attributes = self.data.drop(columns='class').columns
        for i, attribute in enumerate(attributes):
            angle = 2 * np.pi * (i + 0.5) / attribute_count
            label_radius = (circle_radius + 0.3)
            x = label_radius * np.sin(angle)
            y = label_radius * np.cos(angle)
            if 0 <= angle <= np.pi:
                ha = 'left'
            else:
                ha = 'right'
            ax.text(x, y, attribute, ha=ha, va='center', rotation=0, color='darkgrey')
        
        # Style changes: dark grey coordinate axes and labels
        ax.spines['left'].set_color('darkgrey')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)
        plt.xticks(color='darkgrey')
        plt.yticks(color='darkgrey')
        
        # Add legend for class color notation
        for class_name, color in self.color_map.items():
            ax.plot([], [], ' ', label=class_name, marker='o', color=color, markersize=10, markeredgecolor="none")
        ax.legend(loc="best", frameon=False, title="Classes")

        ax.set_aspect('equal')
        plt.show()

def load_and_visualize():
    # Open the file-picker dialog
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select a dataset", filetypes=[("CSV files", "*.csv")])

    if not filepath:
        print("No file selected!")
        return

    # Load the chosen dataset
    df = pd.read_csv(filepath)

    # Check if the dataset has a 'class' column and other columns are numeric
    if 'class' not in df.columns or not all(df.drop(columns='class').applymap(np.isreal).all()):
        print("Invalid dataset! Ensure the dataset has a 'class' column and other columns are numeric.")
        return

    # Visualize the dataset using the SCCWithLegendAndStyle class
    scc_with_legend_and_style = SCCWithChords(df)
    scc_with_legend_and_style.plot()


if __name__ == "__main__":
    load_and_visualize()
