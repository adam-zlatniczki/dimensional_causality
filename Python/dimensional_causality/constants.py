""" Constants for visualization """
import matplotlib

ALPHA = 0.5

"""LaTeX preamble"""
LATEX_SETTINGS = {
    "text.latex.preamble": [
        r'\usepackage{stmaryrd}'
    ]
}
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update(LATEX_SETTINGS)

""" Manifold related constants"""
# Colors
COLOR_X = "red"
COLOR_Y = "blue"
COLOR_J = "black"
COLOR_Z = "gold"

# Labels
LABEL_X = "X"
LABEL_Y = "Y"
LABEL_J = "J"
LABEL_Z = "Z"

""" Causality case constants """
# Colors https://www.w3schools.com/colors/colors_picker.asp
COLOR_X_CAUSES_Y = "red"
COLOR_CIRCULAR_CAUSE = "purple"
COLOR_Y_CAUSES_X = "blue"
COLOR_COMMON_CAUSE = "#f075ae"  # "#EB4291"
COLOR_INDEPENDENCE = "gold"

# Labels
LABEL_X_CAUSES_Y = r"$ \rightarrow $"
LABEL_CIRCULAR_CAUSE = r"$ \leftrightarrow $"
LABEL_Y_CAUSES_X = r"$ \leftarrow $"
LABEL_COMMON_CAUSE = r"$ \curlyveeuparrow $"
LABEL_INDEPENDENCE = r"$ \bot $"

# axis settings
NUM_TICKS = 5  # number of ticks on the x axis
