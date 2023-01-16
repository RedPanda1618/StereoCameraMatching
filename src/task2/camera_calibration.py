import cv2
import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
from scipy import linalg
import json
root = tk.Tk()
root.withdraw()

class CameraCalibration:
    def __init__(self, img_path, photo_rate=1, coordinates_len=6):
        self.photo_rate = photo_rate
        self.img_path = img_path
        self.pprow = None
        self.image_coordinates = []
        self.image_coordinates_flp = []
        self.world_coordinates = []
        self.coordinates_len = 0
        self.img = None
        self.marked_img = None
        self.coordinates_len = coordinates_len
        self.mtx = None
        self.dist = np.zeros(5)
        self.dist_img = None
        self.json_data = None
        self.camera_num = 0

    def get_b(self, world_coordinates, image_coordinates):
        b = np.zeros(((world_coordinates.shape[0] * 2), 11), dtype=np.float64)
        for i in range(len(world_coordinates)):
            b[i*2, 0] = world_coordinates[i, 0]
            b[i*2, 1] = world_coordinates[i, 1]
            b[i*2, 2] = world_coordinates[i, 2]
            b[i*2, 3] = 1
            
            b[i*2+1, 4] = world_coordinates[i, 0]
            b[i*2+1, 5] = world_coordinates[i, 1]
            b[i*2+1, 6] = world_coordinates[i, 2]
            b[i*2+1, 7] = 1

            b[i*2, 8] = -1 * image_coordinates[i, 0] * world_coordinates[i, 0]
            b[i*2, 9] = -1 * image_coordinates[i, 0] * world_coordinates[i, 1]
            b[i*2, 10] = -1 * image_coordinates[i, 0] * world_coordinates[i, 2]

            b[i*2+1, 8] = -1 * image_coordinates[i, 1] * world_coordinates[i, 0]
            b[i*2+1, 9] = -1 * image_coordinates[i, 1] * world_coordinates[i, 1]
            b[i*2+1, 10] = -1 * image_coordinates[i, 1] * world_coordinates[i, 2]
        return b

    def get_q(self):
        q = np.reshape(self.image_coordinates_flp, (self.image_coordinates_flp.shape[0]*2))
        return q

    def select_point(self):
        while(len(self.image_coordinates) < self.coordinates_len):
            window_title = "Choose dots"
            cv2.imshow(window_title, self.marked_img)
            cv2.setMouseCallback(window_title, self.onMouse)
            cv2.waitKey()

    def onMouse(self, event, x_in, y_in, flags, params):
        self.show_pointer(x_in, y_in)
        x = x_in // 2
        y = y_in // 2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.marked_img = self.dist_img.copy()
            self.marked_img = cv2.resize(self.marked_img, (self.json_data["width"]*2, self.json_data["height"]*2))
            for (x_prev, y_prev) in self.image_coordinates:
                cv2.drawMarker(self.marked_img, (int(x_prev)*2, int(y_prev)*2), (255, 255, 0), markerSize=20, markerType=cv2.MARKER_STAR, thickness=2)
            cv2.drawMarker(self.marked_img, (x*2, y*2), (255, 255, 150), markerSize=30, markerType=cv2.MARKER_STAR, thickness=4)
            
            cv2.imshow("Choose dots", self.marked_img)
            
            message_box = "Enter world-coordinates."
            for w_coo in self.world_coordinates:
                message_box += "\n"
                message_box += str(w_coo)
            world_xyz = simpledialog.askstring("Enter coordinates", message_box)
            
            try:
                world_xyx = world_xyz.replace(" ", "").split(",")
                world_x = world_xyx[0]
                world_y = world_xyx[1]
                world_z = world_xyx[2]
                cv2.drawMarker(self.marked_img, (x*2, y*2), (255, 255, 0), markerSize=30, markerType=cv2.MARKER_STAR, thickness=4)
                self.image_coordinates.append([x, y])
                self.world_coordinates.append([float(world_x), float(world_y), float(world_z)])
            except:
                messagebox.showinfo("Attention!", "Canceled this point.")
                self.marked_img = self.dist_img.copy()
                self.marked_img = cv2.resize(self.marked_img, (self.json_data["width"]*2, self.json_data["height"]*2))
                for (x_prev, y_prev) in self.image_coordinates:
                    cv2.drawMarker(self.marked_img, (int(x_prev)*2, int(y_prev)*2), (255, 255, 0), markerSize=20, markerType=cv2.MARKER_STAR, thickness=4)
                    cv2.imshow("Choose dots", self.marked_img)
                return
            if (len(self.image_coordinates) == self.coordinates_len):
                self.image_coordinates = np.asarray(self.image_coordinates, dtype=np.float32)
                self.world_coordinates = np.asarray(self.world_coordinates, dtype=np.float32)
                
                self.image_coordinates_flp = np.zeros_like(self.image_coordinates)
                self.image_coordinates_flp[:, 1] = self.json_data["height"] - self.image_coordinates[:, 1]
                self.image_coordinates_flp[:, 0] = self.image_coordinates[:, 0]
                messagebox.showinfo("Navigation", "Finished selecting coordinates.")
                cv2.destroyAllWindows()
            return
    
    def show_pointer(self, x, y):
        show_size = 10
        if (x < show_size or y < show_size or (x+show_size) > self.json_data["width"]*2 or (y+show_size) > self.json_data["height"]*2):
            return
        pointer_img = self.marked_img[(y-show_size):(y+show_size), (x-show_size):(x+show_size)]
        pointer_img = cv2.resize(pointer_img, (300, 300))
        pointer_img = cv2.drawMarker(pointer_img, (150, 150), (255, 255, 0), cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=1)
        cv2.imshow("pointer position", pointer_img)
        cv2.moveWindow("pointer position", 200, 100)
        
    def calc_pprow(self):
        b = self.get_b(self.world_coordinates, self.image_coordinates_flp)
        q = self.get_q()
        b_t = b.T.copy()
        bt_b = np.dot(b_t, b)
        bt_b_inv = np.linalg.inv(bt_b)
        p = np.dot(np.dot(bt_b_inv, b_t), q)
        p = np.append(p, 1)
        self.pprow = np.reshape(p, (3, 4))
        return self.pprow

    def get_pprow(self):
        if self.pprow is None:
            print("pprow is not calculated.")
            return np.zeros((4, 3))
        return self.pprow
    
    def set_coordinates(self, json_path):
        self.marked_img = self.dist_img.copy()
        self.marked_img = cv2.resize(self.marked_img, (self.json_data["width"]*2, self.json_data["height"]*2))
        for (x_prev, y_prev) in self.image_coordinates:
            cv2.drawMarker(self.marked_img, (int(x_prev)*2, int(y_prev)*2), (255, 255, 0), markerSize=20, markerType=cv2.MARKER_STAR, thickness=2)
        self.select_point()
        self.save_params(json_path)
        self.calc_pprow()
    
    def calc_params(self):
        s = self.pprow[:, :3]
        t = self.pprow[:, 3]
        a, r = linalg.rq(s)
        if ((a[0, 2] < 0) and (a[1, 2] < 0)):
            print("A: adjusted")
            adjuster = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            a = np.dot(a, adjuster)
            r = np.dot(adjuster, r)
        a /= a[2, 2]
        t *= a[2, 2]
        a = np.abs(a)
        if not self.mtx is None:
            a = np.asarray(self.json_data["mtxs"][str(self.camera_num)])
        return a, r, t
        
    def save_params(self, json_path):
        self.json_data["image_coordinates"][str(self.camera_num)] = self.image_coordinates.tolist()
        self.json_data["world_coordinates"][str(self.camera_num)] = self.world_coordinates.tolist()
        with open(json_path, "w") as f:
            js = json.dumps(self.json_data, indent=2)
            f.write(js)
    
    def load_params(self, json_path, camera_num, show=True):
        self.camera_num = camera_num
        self.json_data = self.load_json(json_path)
        
        self.img = cv2.imread(self.img_path)
        try:
            self.dist = np.asanyarray(self.json_data["dists"][str(camera_num)])
            self.mtx = np.asanyarray(self.json_data["mtxs"][str(camera_num)])
            self.get_distortion()
        except:
            self.dist_img = self.img
            self.json_data["width"] = self.img.shape[1]
            self.json_data["height"] = self.img.shape[0]
        
        try:
            self.image_coordinates = np.asarray(self.json_data["image_coordinates"][str(camera_num)][self.img_path], dtype=np.float32)
            self.world_coordinates = np.asarray(self.json_data["world_coordinates"][str(camera_num)][self.img_path], dtype=np.float32)
            if (self.image_coordinates.shape[0] < self.coordinates_len):
                raise ValueError
        except KeyError:
            self.image_coordinates = np.array([])
            self.world_coordinates = np.array([])
            print("Please set " + str(self.coordinates_len) + " coordinates.")
            self.image_coordinates = self.image_coordinates.tolist()
            self.world_coordinates = self.world_coordinates.tolist()
            self.set_coordinates(json_path)
        
        except ValueError:
            print("Please set " + str(self.coordinates_len - self.image_coordinates.shape[0]) + " coordinates.")
            self.image_coordinates = self.image_coordinates.tolist()
            self.world_coordinates = self.world_coordinates.tolist()
            self.set_coordinates(json_path)

        self.image_coordinates_flp = np.zeros_like(self.image_coordinates)
        self.image_coordinates_flp[:, 1] = self.img.shape[0] - self.image_coordinates[:, 1]
        self.image_coordinates_flp[:, 0] = self.image_coordinates[:, 0]
        if show:
            self.show_image_coordinates()
        self.calc_pprow()
    
    def load_json(self, json_path):
        with open(json_path, "r") as f:
            js = f.read()
            self.json_data = json.loads(js)
        return self.json_data
    
    def show_image_coordinates(self):
        img_marked = self.dist_img.copy()
        sorted_points = self.image_coordinates[self.image_coordinates[:, 0].argsort(), :]
        cv2.imshow("Points", self.dist_img)
        cv2.waitKey(50)
        for (x, y) in sorted_points:
            cv2.drawMarker(img_marked, (int(x), int(y)), (255, 255, 150), markerSize=30, markerType=cv2.MARKER_STAR, thickness=2)
            cv2.imshow("Points", img_marked)
            cv2.waitKey(10)
        cv2.imshow("Points", img_marked)
        cv2.waitKey(10)
        cv2.destroyAllWindows()

    def get_distortion(self, img_path=None):
        if img_path is None:
            w, h = self.img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dist_img = cv2.undistort(self.img, self.mtx, self.dist, None, newcameramtx)
            self.dist_img = dist_img
        else:
            img = cv2.imread(img_path)
            w, h = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dist_img = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        
        return dist_img
