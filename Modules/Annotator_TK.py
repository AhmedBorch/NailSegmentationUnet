from Modules.Annotator_imports import *
import tkinter as tk
from tkinter import filedialog, Menu, ttk, messagebox
from os import listdir, scandir, path, remove, rename
import csv
from shutil import copyfile
import pandas as pd


def read_list(line):
    if line == '[[]]':
        return [[]]
    mylist = []
    elems = line[1:-2].split('), ')
    for elem in elems:
        x, y = elem[1:].split(', ')
        mylist.append((int(x), int(y)))
    return mylist

class Annotator:
    """
    Class for the TKinter manager
    """
    def __init__(self):
        self.linked_gui = None
        self.lock = threading.Lock()
        self.root = tk.Tk()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.geometry("600x600")
        self.root.title('Annotator')
        self.root.resizable()
        self.curr_dir = None
        self.files = {}
        self.progress_num = 0

        topframe = tk.Frame(self.root)
        fileframe = tk.Frame(topframe)
        optionsframe = tk.Frame(topframe)
        progressframe = tk.Frame(self.root)

        topframe.pack(expand=1, fill="both")
        fileframe.pack(expand=1, fill="both", side=tk.LEFT)
        optionsframe.pack(expand=1, fill="both", side=tk.LEFT)
        progressframe.pack(expand=1, fill="both", side=tk.BOTTOM)

        # Menu
        menubar = Menu(self.root)
        self.root['menu'] = menubar
        menu_file = Menu(menubar)
        menu_edit = Menu(menubar)
        menubar.add_cascade(menu=menu_file, label='File')
        menubar.add_cascade(menu=menu_edit, label='Edit')
        menu_file.add_command(label='Open...', command=self.open_dir)

        # Fileframe
        scrollbar = tk.Scrollbar(fileframe)
        self.listbox = tk.Listbox(fileframe, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<Double-1>', lambda event: self.open_img())
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.YES)

        self.prev_select = -1
        self.unsaved_changes = 0

        # Options frame
        tabControl = ttk.Notebook(optionsframe)

        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)

        # -------- TAB 1: Image options -------- 

        def create_slider(mst, fr, t, lab, cmd, var, default):
            slider = tk.Scale(mst, from_=fr, to=t, orient=tk.HORIZONTAL, label=lab, command=cmd, variable=var)
            slider.set(default)
            slider.pack()

        self.landmark_slider_var = tk.IntVar()
        landmark_slider_command = lambda event: self.change_image_setting('Marker_visibility', self.landmark_slider_var.get() / 255)
        create_slider(tab1, 0, 255, 'Landmark visibility', landmark_slider_command, self.landmark_slider_var, 255)

        self.landmark_sizer_var = tk.IntVar()
        landmark_sizer_command = lambda event: self.change_image_setting('Marker_size', self.landmark_sizer_var.get())
        create_slider(tab1, 2, 8, 'Landmark size', landmark_sizer_command, self.landmark_sizer_var, 2)

        self.onion_slider_var = tk.IntVar()
        onion_slider_command = lambda event: self.change_image_setting('Onionskin_alpha', self.onion_slider_var.get() / 255)
        create_slider(tab1, 0, 255, 'Onionskin alpha', onion_slider_command, self.onion_slider_var, 255)

        def create_checkbox(master, text, var, func):
            chk = tk.Checkbutton(master, text=text, variable=var, onvalue=1, offvalue=0, command=func)
            chk.pack()

        self.filter = tk.IntVar()
        create_checkbox(tab1, 'Bilateral filtering', self.filter, lambda: self.change_image_setting('Filter', self.filter.get(), self.linked_gui.prepare_image))

        self.unsharp = tk.IntVar()
        create_checkbox(tab1, 'Unsharp masking', self.unsharp, lambda: self.change_image_setting('Unsharp', self.unsharp.get(), self.linked_gui.prepare_image))

        self.histeq = tk.IntVar()
        create_checkbox(tab1, 'CLAHE', self.histeq, lambda: self.change_image_setting('Histogram_eq', self.histeq.get(), self.linked_gui.prepare_image))

        self.gamma_corr = tk.IntVar()
        create_checkbox(tab1, 'Gamma correction', self.gamma_corr, lambda: self.change_image_setting('Gamma_correction', self.gamma_corr.get(), self.linked_gui.prepare_image))

        self.zoom_box = tk.IntVar()
        self.zoom_box.set(1)
        create_checkbox(tab1, 'Zoom box', self.zoom_box, lambda: self.change_image_setting('Zoom_box', self.zoom_box.get(), self.linked_gui.prepare_image))

        self.alpha_slider_var = tk.IntVar()
        alpha_slider_command = lambda event: self.change_image_setting('Alpha_blend', 1.0 - self.alpha_slider_var.get() / 255)
        create_slider(tab1, 0, 255, 'Blend ratio', alpha_slider_command, self.alpha_slider_var, 0)

        scrollbox_frame = tk.Frame(tab1)
        scrollbox_label_frame = tk.Frame(scrollbox_frame)
        scrollbox_button_frame = tk.Frame(scrollbox_frame)
        scrollbox_frame.pack()
        scrollbox_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbox_button_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.disphand_label_var = tk.StringVar(value='Current nail: All')
        self.disphand_label = ttk.Label(scrollbox_label_frame, textvariable=self.disphand_label_var, relief=tk.RAISED)
        self.disphand_label.config(width=17, font=("Courier", 10))
        self.disphand_label.pack(fill=tk.BOTH, expand=True)

        disp_butt_up = tk.Button(scrollbox_frame, text='^',
                                 command=lambda: self.change_hand_num(1))
        disp_butt_down = tk.Button(scrollbox_frame, text='V',
                                   command=lambda: self.change_hand_num(-1))

        disp_butt_up.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        disp_butt_down.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        hand_delete_button = tk.Button(scrollbox_frame, text='X', height=1, command=self.remove_hand)
        hand_delete_button.pack(fill=tk.X)

        # -------- TAB 2: Help -------- 
        
        def create_static_label(master, txt):
            stat_label = ttk.Label(master, text=txt, relief=tk.RAISED)
            stat_label.config(width=17, font=("Courier", 10))
            stat_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        create_static_label(tab2, 'q: Quit')
        create_static_label(tab2, 'w: Save to file')
        create_static_label(tab2, 'n: Get next image (without saving)')
        create_static_label(tab2, 'p: Get previous image (without saving)')
        create_static_label(tab2, 'a: Get next image and save')
        create_static_label(tab2, 's: Save (not to file)')
        create_static_label(tab2, 'r: Reject')
        create_static_label(tab2, 'd: Add another nail')
        create_static_label(tab2, 'Space: Reset markers')

        # -------- TAB 4: export -------- 

        writer_button = tk.Button(tab4, text='Generate training data', height=2, command=self.generate_training_data)
        writer_button.pack(fill=tk.X)

        delete_button = tk.Button(tab4, text='Delete rejected', height=2, command=self.delete_rejected)
        delete_button.pack(fill=tk.X)

        tabControl.add(tab1, text='Options')
        tabControl.add(tab2, text='Help')
        tabControl.add(tab4, text='Export')

        tabControl.pack(expand=1, fill="both")

        self.status_label_var = tk.StringVar()
        self.status_label = ttk.Label(progressframe, textvariable=self.status_label_var, relief=tk.RAISED)
        self.status_label.config(width=200, font=("Courier", 15))
        self.status_label.pack(side=tk.TOP, expand=1, fill='both')

        self.progress_label_var = tk.StringVar()
        self.progress_label = ttk.Label(progressframe, textvariable=self.progress_label_var, relief=tk.RAISED)
        self.progress_label.config(width=200, font=("Courier", 20))
        self.progress_label.pack(side=tk.TOP, expand=1, fill='both')

        self.progress = ttk.Progressbar(progressframe, orient=tk.HORIZONTAL,
                                        length=100, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, expand=1, fill="both")

    def open_dir(self):
        """
        Open a file dialog and select the directory
        containing the col, depth and depthi subfolders, with the optional annotations file
        :return: None
        """
        self.root.withdraw()
        self.files.clear()
        self.listbox.delete(0, tk.END)
        folder_selected = filedialog.askdirectory()
        if folder_selected == '':
            self.root.deiconify()
            return
        subfolders = [f.path.split('\\')[1] for f in scandir(folder_selected) if f.is_dir()]
        if all(sf in subfolders for sf in ['col', 'masks']):
            self.curr_dir = folder_selected
            existing_files = {}
            csvpath = path.join(folder_selected, 'Annotations.csv')
            if path.exists(csvpath):
                with open(csvpath, newline='') as file:
                    reader = csv.reader(file, delimiter=';', quotechar='|')
                    for row in reader:
                        existing_files[row[0]] = [int(row[1])]
                        for temp in row[2:]:
                            existing_files[row[0]].append(read_list(temp))

            all_files = listdir(path.join(folder_selected, 'col'))
            extensions = ('.png', '.jpg', '.jpeg')
            img_files = [f for f in all_files if f.endswith(extensions)]
            for imgf in img_files:
                self.listbox.insert(tk.END, imgf)
                self.files[imgf] = [DATA_EMPTY]
                if imgf in existing_files.keys():
                    self.files[imgf] = existing_files[imgf]
                else:
                    self.files[imgf] = [0, [[]]]
                color = None
                if int(self.files[imgf][0]) == DATA_ANNOT:
                    self.progress_num += 1
                    color = 'green'
                elif int(self.files[imgf][0]) == DATA_REJECTED:
                    self.progress_num += 1
                    color = 'red'
                self.listbox.itemconfig(self.listbox.size() - 1, {'bg': color})
            self.listbox.update()
            self.progress['value'] = round(self.progress_num / len(self.files) * 100)
            self.root.update_idletasks()
            self.progress_label_var.set(
                f'{self.progress_num}/{len(self.files)}, {self.progress_num / len(self.files) * 100:.2f}%')
        else:
            self.status_label_var.set('Incorrect directory structure')
            self.root.update_idletasks()
        self.root.deiconify()

    def open_img(self):
        """
        Open an image when double clicked in the file list.
        :return: None
        """
        selection = self.listbox.curselection()[0]
        fpath = self.listbox.get(selection)
        impath = path.join(self.curr_dir, 'col', fpath)
        data = self.files[fpath]
        landmarks = None
        exists = False
        if data and data[0] == DATA_ANNOT:
            exists = True
            landmarks = data[1:]

        with self.linked_gui.lock:
            if not exists:
                landmarks = [[]]
            self.linked_gui.change_image(impath, landmarks=landmarks)

    def step_elem(self, delta):
        """
        Step in the file list, and open whichever image is the newly selected one.
        :param delta: Number of images to step
        :return: None
        """
        if not self.listbox.curselection():
            selection = -delta
        else:
            selection = self.listbox.curselection()[0]
        self.listbox.selection_clear(0, tk.END)
        newsel = selection + delta
        if -1 < newsel < self.listbox.size():
            self.listbox.select_set(selection + delta)
            self.open_img()
        else:
            self.status_label_var.set(f'Cannot step to selection: {newsel}')
            self.root.update_idletasks()

    def save(self):
        """
        Save the newly made annotation to the internal dictionary (NOT a file!)
        :return: None
        """
        with self.linked_gui.lock:
            landmarks = self.linked_gui.landmarks
        if not self.listbox.curselection():
            return
        selection = self.listbox.curselection()[0]
        self.files[self.listbox.get(selection)] = [DATA_ANNOT] + landmarks
        self.listbox.itemconfig(selection, {'bg': '#33c933'})
        self.unsaved_changes = 1

    def reject(self):
        """
        Save to the internal dictionary that the image is bad (NOT to a file!)
        :return: None
        """
        if not self.listbox.curselection():
            return
        self.status_label_var.set('Rejected')
        selection = self.listbox.curselection()[0]
        self.files[self.listbox.get(selection)] = [DATA_REJECTED]
        self.listbox.itemconfig(selection, {'bg': '#ff6666'})
        self.unsaved_changes = 1

    def save_to_file(self):
        """
        Generate a new annotations.csv file and write the internal dictionary to it.
        :return: None
        """
        if not self.files:
            return
        csvpath = path.join(self.curr_dir, 'Annotations_temp.csv')
        self.progress_num = 0
        with open(csvpath, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';', quotechar='|')
            for idx, (f, d) in enumerate(self.files.items()):
                if d:
                    if d[0] != DATA_EMPTY:
                        self.progress_num += 1
                    row = [f, d[0]]
                    for temp in d[1:]:
                        row.append(temp)
                    #row = [f, d[0], d[1][0]]
                    writer.writerow(row)
                    color = None
                    if d[0] == DATA_ANNOT:
                        color = 'green'
                    elif d[0] == DATA_REJECTED:
                        color = 'red'
                    self.listbox.itemconfig(idx, {'bg': color})
                    with self.linked_gui.lock:
                        mask = polygon_mask((self.linked_gui.image.shape[0], self.linked_gui.image.shape[1]), d[1:])
                    cv2.imwrite(path.join(self.curr_dir, 'masks', f), mask*255)
        old_file = path.join(self.curr_dir, 'Annotations.csv')
        # Overwrite the old file with the new one
        # This ensures that data is not lost if the annotator crashes on trying to open the old file
        if path.exists(old_file):
            remove(old_file)
        rename(csvpath, old_file)
        self.progress['value'] = round(self.progress_num / len(self.files) * 100)
        self.progress_label_var.set(
            f'{self.progress_num}/{len(self.files)}, {self.progress_num / len(self.files) * 100:.2f}%')
        self.status_label_var.set('Saved to file')
        self.root.update_idletasks()
        self.unsaved_changes = 0

    def change_image_setting(self, varname, newval, on_done=None):
        """
        Change one of the image parameters in the linked GUI in a synchronous manner
        :param varname: Variable to change
        :param newval: New value of the variable
        :param on_done: A callback to perform on changing the variable. Usually linked_gui.prepare_image()
        :return: None
        """
        with self.linked_gui.lock:
            self.linked_gui.image_params[varname] = newval
        if on_done and self.linked_gui.image is not None:
            on_done()

    def get_image_setting(self, varname):
        """
        Get one of the image parameters in the linked GUI in a synchronous manner
        :param varname: Variable to get
        :return: Variable value
        """
        with self.linked_gui.lock:
            return self.linked_gui.image_params[varname]

    def change_hand_num(self, delta):
        """
        Change the hand currently on display
        :param delta: Number of hands to step
        :return: None
        """
        with self.linked_gui.lock:
            disp_hand = max(-1, min(self.linked_gui.image_params['Display_nail'] + delta,
                                    (len(self.linked_gui.landmarks)) - 1))
            self.linked_gui.image_params['Display_nail'] = disp_hand
            self.linked_gui.linked_points.clear()
            self.linked_gui.select_rect = None
            self.linked_gui.mouse_data = None
            self.linked_gui.extra_data = None
            self.linked_gui.orig_points.clear()
            self.linked_gui.nextmode = MOVE
        self.disphand_label_var.set(f'Current hand: {disp_hand if disp_hand != -1 else "ALL"}')

    def remove_hand(self):
        """
        Remove the currently displayed hand
        :return: None
        """
        with self.linked_gui.lock:
            current_hand = self.linked_gui.image_params['Display_nail']
            hand_num = len(self.linked_gui.landmarks)
            if current_hand == -1 or hand_num == 1:
                return
            del self.linked_gui.landmarks[current_hand]
            self.linked_gui.image_params['Display_nail'] = -1
            self.linked_gui.mouse_data = None
            self.linked_gui.extra_data = None
            self.linked_gui.nextmode = MOVE
            self.linked_gui.update_mask()
        self.disphand_label_var.set('Current nail: ALL')

    def generate_training_data(self):
        """
        Export the contents of Annotations.csv to a new file with only the minimum necessary data.
        (i.e. without the rejected images, the status marker, etc.)
        :return: None
        """
        if not self.files:
            return
        csvpath = path.join(self.curr_dir, 'Annotations.csv')
        trainpath = path.join(self.curr_dir, 'Training_data.csv')
        with open(csvpath, 'r', newline='') as ann, open(trainpath, 'w', newline='') as train:
            reader = csv.reader(ann, delimiter=';', quotechar='|')
            writer = csv.writer(train, delimiter=';', quotechar='|')
            for row in reader:
                if row[1] == '1':
                    writer.writerow([row[0]] + row[2:])
        self.status_label_var.set('Training file generated')

    def delete_rejected(self):
        """
        Delete all the rejected images from the file system
        :return: None
        """
        if not self.files or len(self.files) == 0:
            return

        message = "You're about to delete all rejected images. This operation cannot be reversed. Are you sure?"
        confirm = messagebox.askyesno(
            title="Save On Close",
            message=message,
            default=messagebox.NO,
            parent=self.root)
        if confirm is None or not confirm:
            return
        csvpath = path.join(self.curr_dir, 'Annotations.csv')
        self.progress_num = 0
        to_remove_keys = []
        to_remove_idxs = []
        with open(csvpath, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';', quotechar='|')
            for idx, (f, d) in enumerate(self.files.items()):
                if d:
                    if d[0] == -1:
                        to_remove_keys.append(f)
                        to_remove_idxs.append(idx)
                        for rempath in [path.join(self.curr_dir, 'col', f), path.join(self.curr_dir, 'depthi', f), path.join(self.curr_dir, 'depth', f)]:
                            remove(rempath)
                        continue
                    if d[0] == 1:
                        self.progress_num += 1
                    writer.writerow([f] + d)
                    color = None
                    if d[0] == 1:
                        color = 'green'
                    self.listbox.itemconfig(idx, {'bg': color})

        for k in to_remove_keys:
            self.files.pop(k)

        for idx in reversed(to_remove_idxs):
            self.listbox.delete(idx)

        self.listbox.selection_clear(0, tk.END)
        self.listbox.select_set(0)

        self.progress['value'] = round(self.progress_num / len(self.files) * 100)
        self.progress_label_var.set(
            f'{self.progress_num}/{len(self.files)}, {self.progress_num / len(self.files) * 100:.2f}%')
        self.root.update_idletasks()
        self.unsaved_changes = 0
        self.status_label_var.set('Rejected images deleted')

        self.open_img()

    def on_close(self):
        """
        A callback for when either window is closed.
        If there are any unsaved changes, ask the user if they want to quit or save.
        :return: None
        """
        if self.unsaved_changes == 1:
            message = 'You have unsaved changes. Would you like to save before exiting?'
            confirm = messagebox.askyesnocancel(
                title="Save On Close",
                message=message,
                default=messagebox.YES,
                parent=self.root)
            if confirm:
                self.save_to_file()
            elif confirm is None:
                return
        self.root.quit()
        with self.linked_gui.lock:
            self.linked_gui.alive = False