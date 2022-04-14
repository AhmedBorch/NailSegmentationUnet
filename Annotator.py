from Modules.Annotator_CV import *
from Modules.Annotator_TK import *

# ---------- How to use: ----------
# Run python Annotator.py
# File -> open...
# Select a folder with the col and masks subfolders.
# Double click on an image in the file list to open it.
# Click anywhere to add vertices.
# The first vertex is blue, the last is red. If you add a new vertex, it will connect to these two.
# You can drag existing vertices with the left mouse button or delete them with the right.
# You can use the blend ratio slider to check the generated mask.
# Press d to add a new polygon. Use the current nail scrollbox to cycle through nails and use the x button to delete the current one.
# (You will always add new vertices to the current polygon, or the last one if all are visible)
# See the help menu for more button commands
# Press s or a to save, then w to save to file.


gui = GUI()  # Global variable for GUI.


def TKloop():
    """
    TKinter main loop. Should be ran in a thread separate from the main or the OpenCV ones.
    :return: None
    """
    annot = Annotator()
    annot.linked_gui = gui
    gui.linked_annot = annot
    annot.root.mainloop()


if __name__ == "__main__":
    annot_thread = threading.Thread(name='Annotator', target=TKloop)
    GUI_thread = threading.Thread(name='GUI', target=gui.start)
    annot_thread.start()
    GUI_thread.start()
    annot_thread.join()
    GUI_thread.join()
    quit()
