import cv2.cv2
import numpy as np
from Modules.Annotator_imports import *
import win32api
import win32con
import pyrealsense2 as rs


def coord_add(tup1, tup2, alpha=1, beta=1):
    """
    Add a pair of coordinates. You can also specify a coefficient for x and y each.
    :param tup1: First x-y pair
    :param tup2: Second x-y pair
    :param alpha: Coefficient for second x
    :param beta: Coefficient for second y
    :return: Resulting coordinates
    """
    return tup1[0] + int(tup2[0] * alpha), tup1[1] + int(tup2[1] * beta)


def clip_coords(c, xmin=0, xmax=IMSIZEX - 1, ymin=0, ymax=IMSIZEY - 1):
    """
    Clip coordiante values to stay within the window
    :param c: Coordiantes
    :param xmin: x minimum value
    :param xmax: x maximum value
    :param ymin: y minimum value
    :param ymax: y maximum value
    :return: Clipped coordiantes
    """
    return np.clip(c[0], xmin, xmax), np.clip(c[1], ymin, ymax)


# https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0138-1
def gamma_correction(img, tau=3):
    V = img[:, :, 2]
    mu = np.mean(V)
    sigma = np.std(V)
    gamma = -np.log2(sigma) if 4 * sigma <= (1 / tau) else np.exp((1 - (mu + sigma)) / 2)
    Ig = np.power(V, gamma)
    c = 1
    if mu <= 0.5:
        k = Ig + (1 - Ig) * (mu ** gamma)
        c = 1 / (1 + (k - 1))
    Io = c * Ig
    img[:, :, 2] = Io
    return img.copy()


class GUI:
    """
    Class for OpenCV GUI
    """

    def __init__(self):
        self.image = cv2.resize(cv2.imread('hand_small.jpg'), (192, 192))
        self.col_image = self.image
        self.blend_image = self.image
        self.mask = np.zeros((self.col_image.shape[0], self.col_image.shape[1]), np.uint8)
        self.small_image_size_x = 48
        self.small_image_size_y = 48
        self.small_image_zoom = 2
        self.small_image = cv2.resize(self.image, (self.small_image_size_y, self.small_image_size_x))
        self.small_image_loc = False
        self.small_image_center = (96, 96)
        self.lock = threading.Lock()
        self.image_params = {
            'Alpha_blend': 1.0,
            'Marker_visibility': 1.0,
            'Histogram_eq': False,
            'Marker_size': 2,
            'Display_nail': -1,
            'Gamma_correction': False,
            'Filter': False,
            'Unsharp': False,
            'Zoom_box': True,
        }
        self.linked_annot = None  # The TKinter file manager

        # Old annotator ---------------
        self.linked_points = []
        self.orig_points = []
        self.mode = MOVE
        self.nextmode = MOVE
        self.mouse_data = None
        self.extra_data = None
        self.select_rect = None
        self.cursor_type = win32con.IDC_CROSS
        self.alive = True
        self.ctrldown = False

        self.landmarks = [[]]
        self.onion_prev, self.onion_next = [], []
        
        self.reset()

    # Old annotator ---------------

    def update_mask(self):
        self.mask = polygon_mask(self.mask.shape, self.landmarks)

    def mouse_callback(self, event, x, y, flags, param):
        """
        OpenCV mouse callback for every mouse event within the annotation window
        :param event: Mouse event type
        :param x: Event coordinate x
        :param y: Event coordiante y
        :param flags: Mouse event flags (unused)
        :param param: Mouse eventparams (unused)
        :return: None
        """

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                newval = self.small_image_zoom*2
            else:
                newval = self.small_image_zoom // 2
            self.small_image_zoom = max(2, min(newval, 8))

        distance_thresh = max(self.image_params['Marker_size'] ** 2, 10)
        if self.nextmode is not None or self.image_params['Marker_visibility'] < 1. / 255 or self.image is None:
            return
        if self.mode == MOVE:
            if event == cv2.EVENT_LBUTTONDOWN or (flags == cv2.EVENT_FLAG_CTRLKEY and not self.ctrldown):
                if flags == cv2.EVENT_FLAG_CTRLKEY:
                    self.ctrldown = True
                mindist = -1
                for lms, active_landmarks in enumerate(self.landmarks):
                    if self.image_params['Display_nail'] != -1 and lms != self.image_params['Display_nail']:
                        continue
                    for i, l in enumerate(active_landmarks):
                        distsqr = (l[0] - x) ** 2 + (l[1] - y) ** 2
                        if distsqr < distance_thresh and (distsqr > mindist or mindist == -1):
                            mindist = distsqr
                            self.mouse_data = i
                            self.extra_data = lms
                            # Find marker closest to mouse and save to mouse data
                if self.mouse_data is None:
                    self.landmarks[self.image_params['Display_nail']].append((x, y))
                    self.extra_data = -1
                    self.update_mask()
                else:
                    # If user clicked on a landmark
                    self.select_rect = None
                    if self.cursor_type != win32con.IDC_SIZEALL:
                        win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEALL))
                        self.cursor_type = win32con.IDC_SIZEALL

            if event == cv2.EVENT_LBUTTONUP or (flags != cv2.EVENT_FLAG_CTRLKEY and self.ctrldown):
                # If user lets go of a landmark
                if self.ctrldown:
                    self.ctrldown = False
                self.mouse_data = None
                # self.linked_points.clear()
                if self.cursor_type != win32con.IDC_CROSS:
                    win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_CROSS))
                    self.cursor_type = win32con.IDC_CROSS

            if event == cv2.EVENT_MOUSEMOVE or self.ctrldown:
                # If user moves a landmark
                if self.mouse_data is not None:
                    self.landmarks[self.extra_data][self.mouse_data] = (x, y)
                # General move event
                self.small_image_center = (x, y)

            if event == cv2.EVENT_RBUTTONDOWN:
                # Disable landmark
                mindist = -1
                clicked = -1
                for lms, active_landmarks in enumerate(self.landmarks):
                    if self.image_params['Display_nail'] != -1 and lms != self.image_params['Display_nail']:
                        continue
                    for i, l in enumerate(active_landmarks):
                        distsqr = (l[0] - x) ** 2 + (l[1] - y) ** 2
                        if distsqr < distance_thresh and (distsqr > mindist or mindist == -1):
                            mindist = distsqr
                            clicked = i
                            self.extra_data = lms
                if clicked != -1:
                    del self.landmarks[self.extra_data][clicked]
                    self.update_mask()

            if event == cv2.EVENT_LBUTTONUP:
                # If user lets go of the rect
                self.extra_data = -1
                self.mouse_data = None
                self.select_rect = None
                self.linked_points.clear()
                self.orig_points.clear()
                self.nextmode = MOVE
                self.update_mask()

    def flip(self):
        """
        Flip the selected landmarks around their vertical midline
        :return: None
        """
        if not self.linked_points:
            return
        aidx_start = self.image_params['Display_nail'] * 21 if self.image_params['Display_nail'] != -1 else 0
        aidx_end = (self.image_params['Display_nail'] + 1) * 21 if self.image_params['Display_nail'] != -1 else None
        active_landmarks= self.landmarks[aidx_start:aidx_end]

        x_center = np.sum([active_landmarks[l][0] for l in self.linked_points]) / len(self.linked_points)
        y_center = np.sum([active_landmarks[l][1] for l in self.linked_points]) / len(self.linked_points)

        for l in self.linked_points:
            cent = coord_add(active_landmarks[l], (x_center, y_center), -1, -1)
            active_landmarks[l] = coord_add((-1 * cent[0], cent[1]), (x_center, y_center))

        self.landmarks[aidx_start:aidx_end] = active_landmarks

    def reset(self):
        """
        Return all landmarks tot heir original configuration
        :return: None
        """
        self.image_params['Display_nail'] = -1
        if self.linked_annot is not None:
            with self.linked_annot.lock:
                self.linked_annot.disphand_label_var.set('Current hand: ALL')

    def draw(self, im):
        """
        Draw landmarks, onionskin, selection recangles, etc.
        :param im: The original, blended iamge
        :return: The image with everything drawn on
        """
        retim = im
        if self.image_params['Marker_visibility'] > 1. / 255:
            # Drawing markers
            orig = im.copy()
            im = im.copy()
            #print(self.landmarks)
            for nail, landmarks in enumerate(self.landmarks):
                if self.image_params['Display_nail'] != -1 and nail != self.image_params['Display_nail']:
                    continue
                for i in range(len(landmarks)-1):
                    cv2.line(im, landmarks[i], landmarks[i+1], (0, 255, 0), self.image_params['Marker_size'] // 2)
                if len(landmarks) > 1:
                    cv2.line(im, landmarks[0], landmarks[-1], (0, 255, 0), self.image_params['Marker_size'] // 2)
                for idx, l in enumerate(landmarks):
                    color = (200, 0, 200)
                    if idx == 0:
                        color = (200, 0, 0)
                    elif idx == len(landmarks)-1:
                        color = (0, 0, 200)
                    cv2.circle(im, l, self.image_params['Marker_size'], color, self.image_params['Marker_size'] // 2)

            alpha = self.image_params['Marker_visibility']
            final = cv2.addWeighted(im, alpha, orig, 1 - alpha, 0)
            retim = cv2.addWeighted(final, self.image_params['Alpha_blend'], cv2.cvtColor(self.mask * 255, cv2.COLOR_GRAY2BGR), (1 - self.image_params['Alpha_blend']), 0)

            if self.image_params['Zoom_box']:
                small_image_size_x = self.small_image_size_x // self.small_image_zoom
                small_image_size_y = self.small_image_size_y // self.small_image_zoom
                self.small_image = np.zeros((small_image_size_y, small_image_size_x, 3))
                xstart, xend = self.small_image_center[0] - small_image_size_x // 2, self.small_image_center[0] + small_image_size_x // 2
                ystart, yend = self.small_image_center[1] - small_image_size_y // 2, self.small_image_center[1] + small_image_size_y // 2
                xstart_neg, xend_neg = (xstart < 0) * xstart, (xend >= self.image.shape[1]) * (xend - self.image.shape[1])
                ystart_neg, yend_neg = (ystart < 0) * ystart, (yend >= self.image.shape[0]) * (yend - self.image.shape[0])
                sixs, sixe = -xstart_neg, small_image_size_x - xend_neg
                siys, siye = -ystart_neg, small_image_size_y - yend_neg
                rixs, rixe = xstart - xstart_neg, xend - xend_neg
                riys, riye = ystart - ystart_neg, yend - yend_neg
                self.small_image[siys:siye, sixs:sixe] = retim[riys:riye, rixs:rixe]
                self.small_image_loc = self.small_image_center[0] > self.image.shape[1] // 2 and\
                                       self.small_image_center[1] < self.image.shape[0] // 2
                cv2.circle(self.small_image, (small_image_size_x // 2, small_image_size_y // 2), 1, (0, 0, 255))
                start = None if self.small_image_loc else -self.small_image_zoom * small_image_size_x
                end = self.small_image_zoom * small_image_size_x if self.small_image_loc else None
                retim[:self.small_image_zoom * small_image_size_y, start:end] = cv2.resize(self.small_image, (0, 0), fx=self.small_image_zoom, fy=self.small_image_zoom)

        return retim

    def prepare_image(self):
        """
        Perform image processing options that don't need to be done every iteration
        :return: None
        """
        colim = self.col_image
        if self.image_params['Filter']:
            colim = cv2.bilateralFilter(colim, 13, 75, 1)
        if self.image_params['Unsharp']:
            unsharp_mask = cv2.GaussianBlur(colim, (0, 0), 5)
            colim = cv2.addWeighted(colim, 1.5, unsharp_mask, -0.5, 0)
        if self.image_params['Gamma_correction']:
            colim = cv2.cvtColor(colim, cv2.COLOR_BGR2HSV)/255
            colim = gamma_correction(colim)
            colim = cv2.cvtColor((colim*255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.image_params['Histogram_eq']:
            colim = cv2.cvtColor(colim, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE()
            colim[:, :, 2] = clahe.apply(colim[:, :, 2])
            colim = cv2.cvtColor(colim, cv2.COLOR_HSV2BGR)
        self.image = colim

    def change_image(self, colpath, landmarks=None):
        """
        Switch to a new image as specified by TKinter
        :param colpath: Colour image path
        :param depth_path: Depth image path
        :param landmarks: Old landmarks, if we want to load any
        :param landmarks_exist: True if the landmarks are loaded from annotations.csv, false otherwise
        :return:
        """
        self.col_image = cv2.imread(colpath)
        self.image = self.col_image
        self.small_image_size_x = self.image.shape[1] // 2
        self.small_image_size_y = self.image.shape[0] // 2
        self.small_image_zoom = 2
        self.prepare_image()
        self.landmarks = [[]]
        self.mask = np.zeros((self.col_image.shape[0], self.col_image.shape[1]), np.uint8)
        if landmarks is not None and landmarks != [[]]:
            self.landmarks = landmarks
            self.update_mask()
        else:
            self.reset()

    def event_handler(self, key):
        """
        Keystroke event handler for the cv2.waitKey function
        :param key: Pressed key (if any)
        :return: None
        """
        if key == ord('q'):  # Quit
            self.lock.release()
            retval = self.linked_annot.on_close()
            self.lock.acquire()
            return retval

        if key == ord('w'):  # Save and don't quit
            self.lock.release()
            self.linked_annot.save_to_file()
            self.lock.acquire()

        if key == ord('n'):  # Next
            self.lock.release()
            self.linked_annot.step_elem(1)
            self.lock.acquire()

        if key == ord('p'):  # Prev
            self.lock.release()
            self.linked_annot.step_elem(-1)
            self.lock.acquire()

        if key == ord('s'):  # Save
            self.lock.release()
            self.linked_annot.save()
            self.lock.acquire()

        if key == ord('a'):  # Save and next (Advance)
            self.lock.release()
            self.linked_annot.save()
            self.linked_annot.step_elem(1)
            self.lock.acquire()

        if key == ord('d'):
            self.landmarks.append([])
            self.image_params['Display_nail'] = -1
            with self.linked_annot.lock:
                self.linked_annot.disphand_label_var.set('Current hand: ALL')

        if key == ord('r'):  # Reject
            self.lock.release()
            self.linked_annot.reject()
            self.linked_annot.step_elem(1)
            self.lock.acquire()

        if key == ord('f'):  # Flip
            self.flip()

        if key == ord(' '):  # Reset
            self.reset()

        return False

    def start(self):
        """
        Main loop for the OpenCV GUI. Should be ran as a thread separate from main or the TKinter one
        :return: None
        """
        cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotator', 768, 768)
        cv2.setMouseCallback('Annotator', self.mouse_callback)
        self.lock.acquire()
        while self.alive:
            if self.image is None:
                # If the image isn't loaded yet, don't do anything
                self.lock.release()
                cv2.waitKey(30)
                self.lock.acquire()
                continue
            if cv2.getWindowProperty('Annotator', 0) < 0:
                # If the window is closed, try to quit
                with self.linked_annot.lock:
                    self.lock.release()
                    retval = self.linked_annot.on_close()
                    self.lock.acquire()
                    if retval:
                        break
                    else:
                        cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Annotator', 848, 480)
                        cv2.setMouseCallback('Annotator',
                                             self.mouse_callback if self.global_mode == GLOBAL_ANNOT else self.mouse_callback_order)
            cv2.imshow('Annotator', self.draw(self.image))
            self.lock.release()
            key = cv2.waitKey(30)
            self.lock.acquire()
            stop = self.event_handler(key)
            if self.nextmode is not None:
                self.mode = self.nextmode
                self.nextmode = None
            if stop:
                break
        cv2.destroyAllWindows()
