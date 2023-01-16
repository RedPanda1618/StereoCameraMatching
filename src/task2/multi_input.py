import cv2
from cv2 import aruco
from datetime import datetime
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
import pyrealsense2 as rs
import os
import json
import shutil
import numpy as np
root = tk.Tk()
root.withdraw()

class MultiInput:
    def __init__(self, cameras_num, width=640, height=480, intr=False):
        self.save_dir = "data/MultiCamera"
        os.makedirs(self.save_dir, exist_ok=True)
        self.cameras_num = cameras_num
        self.width = width
        self.height = height
        self.view_fps = 6
        self.intr = intr
        self.dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        self.geom_added = False
        self.marker_num = 18
        self.marker_coordinates = [[6.5, 0, 3],
                                   [16, 0, 3],
                                   [25.5, 0, 3],
                                   [6.5, 0, 12.5],
                                   [16, 0, 12.5],
                                   [25.5, 0, 12.5],
                                   [3, 3, 0],
                                   [12.5, 3, 0],
                                   [22, 3, 0],
                                   [3, 12.5, 0],
                                   [12.5, 12.5, 0],
                                   [22, 12.5, 0],
                                   [0, 3, 3],
                                   [0, 12.5, 3],
                                   [0, 22, 3],
                                   [0, 3, 12.5],
                                   [0, 12.5, 12.5],
                                   [0, 22, 12.5]]
    
    def capture(self):
        self.folder_name = os.path.join(self.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.json_data = {}
        self.json_data["dists"] = {}
        self.json_data["mtxs"] = {}
        self.json_data["calibration_img_paths"] = {}
        self.json_data["img_paths"] = []
        self.json_data["marker_ids"] = {}
        self.json_data["image_coordinates"] = {}
        self.json_data["world_coordinates"] = {}
        self.json_data["epipolor_line"] = []
        
        if self.intr:
            for i in range(self.cameras_num):
                intr = self.get_intrinsics()
                self.save_intr(intr, i)
                
                if i < (self.cameras_num - 1):
                    messagebox.showinfo("案内", str(i) + "のカメラ終了。次に切り替えてください。")
                else:
                    messagebox.showinfo("案内", "終了しました。")
        
        self.show_all_camera()
        cnt = self.capture_cv2()
        json_path = self.folder_name + "/params.json"
        if (cnt > 0):
            if not os.path.exists(self.folder_name):
                os.mkdir(self.folder_name)
            js = json.dumps(self.json_data, indent=2)
            with open(json_path, "w", encoding="utf8") as f:
                f.write(js)
        elif (os.path.exists(self.folder_name)):
            shutil.rmtree(self.folder_name)
        return json_path

    def data_format(self, img_dir, photo_num, mode):
        self.folder_name = self.save_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.json_data = {}
        self.json_data["dists"] = {}
        self.json_data["mtxs"] = {}
        self.json_data["calibration_img_paths"] = {}
        self.json_data["img_paths"] = []
        self.json_data["marker_ids"] = {}
        self.json_data["image_coordinates"] = {}
        self.json_data["world_coordinates"] = {}
        for i in range(photo_num):
            frames = []
            marked_frames = []
            marker_ids_list = []
            corners_list = []
            for num in range(self.cameras_num):
                if (mode == 0):
                    img_path = os.path.join(img_dir, "multi_" + str(i) + "_camera" + str(num) + ".png")
                elif (mode == 1):
                    img_path = os.path.join(img_dir, str(format(i, "02")) + "_" + str(num) + ".png")
                else:
                    print("select correct mode.")
                    return
                if os.path.exists(img_path):
                    frame = cv2.imread(img_path)
                    frame = cv2.resize(frame, (self.width, self.height))
                    marked_frame, marker_ids, corners = self.search_mark(frame)
                    marked_frames.append(marked_frame)
                    marker_ids_list.append(marker_ids)
                    corners_list.append(corners)
                else:
                    print(img_path + " is not exist.")
                    marker_ids_list.append([])
                    marked_frames.append([])
                    corners_list.append([])
                    frame = []
                    
                frames.append(frame)
            self.save_data_cv2(frames, i, self.folder_name, corners_list, marker_ids_list)
        
        json_path = self.folder_name + "/params.json"
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
        js = json.dumps(self.json_data, indent=2)
        with open(json_path, "w", encoding="utf8") as f:
            f.write(js)
    
    def capture_pyrealsence2(self):
        cnt = 0
        while True:
            for i in range(self.cameras_num):
                if (cnt == 0):
                    config = rs.config()
                    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.view_fps)
                    pipeline = rs.pipeline()
                    profile = pipeline.start(config)
                    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                else:
                    color_intr = None
                
                while True:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imshow("camera num: "+str(i), color_image)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        return
                    elif key == ord("s"):
                        json_data = self.save_data(color_image, cnt, i, self.folder_name, color_intr, json_data)
                        break
                    
                cv2.destroyAllWindows()
                pipeline.stop()
                if i < (self.cameras_num - 1):
                    messagebox.showinfo("案内", "次のカメラに切り替えてください。")
            
            cnt += 1
            res = messagebox.askquestion("確認", "もう一巡してカメラで写真を撮りますか。")
            if (res == "yes"):
                continue
            else:
                break
        return cnt, json_data
    
    def capture_cv2(self):
        cnt = 0
        cap_list = {}
        for num in self.camera_nums:
            cap = cv2.VideoCapture(self.camera_list[str(num)])
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap_list[str(num)] = cap
            
        while True:
            frames = []
            marked_frames = []
            marker_ids_list = []
            corners_list = []
            for num in self.camera_nums:
                cap = cap_list[str(num)]
                ret, frame = cap.read()
                if ret:
                    marked_frame, marker_ids, corners = self.search_mark(frame)
                    marked_frame = cv2.resize(marked_frame, (marked_frame.shape[1]//2, marked_frame.shape[0]//2))
                    marked_frames.append(marked_frame)
                    marker_ids_list.append(marker_ids)
                    corners_list.append(corners)
                else:
                    marked_frames.append(np.zeros((self.height, self.width, 3)))
                    marker_ids_list.append([])
                    corners_list.append([])
                    frame = np.zeros((self.height, self.width, 3))
                    break
                frames.append(frame)
            tiles = cv2.hconcat(marked_frames)
            save_marker = True
            for i in range(self.cameras_num):
                if (len(marker_ids_list[i]) != self.marker_num):
                    save_marker = False
            save_marker = False
            cv2.imshow("Capture", tiles)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif (key == ord("s")) or save_marker:
                self.save_data_cv2(frames, cnt, self.folder_name, corners_list, marker_ids_list)
                cnt += 1
        cv2.destroyAllWindows()
        for num in self.camera_nums:
            cap = cap_list[str(num)]
            cap.release()
        return cnt
    
    def show_all_camera(self):
        camera_list = []
        self.camera_list = {}
        for camera_number in range(0, 9):
            cap = cv2.VideoCapture(camera_number)
            ret, frame = cap.read()
            if ret is True:
                camera_list.append(camera_number)

        camera_nums = []
        for num in camera_list:
            cap = cv2.VideoCapture(num)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("camera num", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
            res = messagebox.askquestion("", "Do you use this camera?")

            if (res == "yes"):
                camera_num = int(simpledialog.askstring("Enter camera number", "What the number of this camera?"))
                self.camera_list[str(camera_num)] = num
                camera_nums.append(camera_num)
        camera_nums = np.asarray(camera_nums)
        self.camera_nums = np.sort(camera_nums)
            
        cv2.destroyAllWindows()
    
    def save_data(self, color_image, cnt, camera_num, folder_name, color_intrinsict, json_data):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            
        if (cnt == 0):
            self.save_intr(color_intrinsict, json_data)
        
        img_path = folder_name + "/stereo_" + str(cnt) + "_camera" + str(camera_num) + ".png"
        cv2.imwrite(img_path, color_image)
        
        if (camera_num == 0):
            json_data["img_paths"].append([])
        json_data["img_paths"][-1].append(img_path)
        
        return json_data
    
    def save_data_cv2(self, color_images, cnt, folder_name, corners_list, marker_ids_list):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        save_marker = [True] * self.cameras_num
        save_marker_cnt = 0
        pair = ""
        for i in range(self.cameras_num):
            if (len(marker_ids_list[i]) != self.marker_num):
                save_marker[i] = False
            else:
                pair += str(i) + ","
                save_marker_cnt += 1
        if (save_marker_cnt > 1):
            if (pair not in self.json_data["calibration_img_paths"]):
                self.json_data["calibration_img_paths"][pair] = []
            self.json_data["calibration_img_paths"][pair].append([])
        else:
            save_marker = [False] * self.cameras_num
        
        
        for camera_num in range(self.cameras_num):
            if (camera_num == 0):
                self.json_data["img_paths"].append([])
            if (len(color_images[camera_num]) == 0):
                continue
            img_path = folder_name + "/multi_" + str(cnt) + "_camera" + str(camera_num) + ".png"
            cv2.imwrite(img_path, color_images[camera_num])
            print("saved [%s]" %img_path)
            
            self.json_data["img_paths"][-1].append(img_path)
            if save_marker[camera_num]:
                print("saving coordinates")
                try:
                    self.json_data["calibration_img_paths"][pair][-1].append(img_path)
                except:
                    print(pair)
                    print(save_marker_cnt)
                    exit()
                if (str(camera_num) not in self.json_data["image_coordinates"]):
                    self.json_data["marker_ids"][str(camera_num)] = []
                    self.json_data["image_coordinates"][str(camera_num)] = {}
                    self.json_data["world_coordinates"][str(camera_num)] = {}
                self.json_data["marker_ids"][str(camera_num)].append(marker_ids_list[camera_num])
                self.json_data["world_coordinates"][str(camera_num)][img_path] = []
                for marker_id in marker_ids_list[camera_num]:
                    self.json_data["world_coordinates"][str(camera_num)][img_path].append(self.marker_coordinates[marker_id])
                self.json_data["image_coordinates"][str(camera_num)][img_path]= corners_list[camera_num]
    
    def save_intr(self, color_intrinsict, camera_num):
        dist = color_intrinsict.coeffs
        ppx = color_intrinsict.ppx
        ppy = color_intrinsict.ppy
        fx = color_intrinsict.fx
        fy = color_intrinsict.fy
        mtx = np.zeros((3, 3), dtype=np.float32)
        mtx[0, 0] = fx
        mtx[1, 1] = fy
        mtx[0, 2] = ppx
        mtx[1, 2] = ppy
        mtx[2, 2] = 1
        mtx = mtx.tolist()
        height = color_intrinsict.height
        width = color_intrinsict.width
        self.json_data["dists"][str(camera_num)] = dist
        self.json_data["mtxs"][str(camera_num)] = mtx
        self.json_data["height"] = height
        self.json_data["width"] = width
    
    def get_intrinsics(self):
        conf = rs.config()
        conf.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 6)
        pipe = rs.pipeline()
        profile = pipe.start(conf)
        intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
        pipe.stop()
        return intr
    
    def search_mark(self, color_image):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, self.dict_aruco, parameters=self.parameters)
        marked_frame = aruco.drawDetectedMarkers(color_image.copy(), corners, ids)
        id_list = np.ravel(ids).tolist()
        corners = np.asarray(corners, dtype=np.int32).tolist()
        if (len(corners) > 0):
            marker_list = []
            for co in corners:
                marker_point = co[0][0]
                marker_list.append(marker_point)
            return marked_frame, id_list, marker_list, 
        else:
            return color_image, [], []
        
    def load_json(self, json_path):
        with open(json_path, "w") as f:
            json_data = json.loads(f)
        return json_data
    