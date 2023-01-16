from camera_calibration import CameraCalibration
import cv2
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from datetime import datetime
import json
import pickle
import reconstruction3d as r3d
try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False

class StereoVision:
    def __init__(self, json_path, camera_list, bg_num, coordinates_len=6, window_size=3, cpu_rate=0.4, mode=2):
        self.json_path = json_path
        self.home_path = json_path[:json_path.rfind("/")]
        self.save_path = self.home_path + "/calculated"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        with open(self.json_path, "r") as f:
            js = f.read()
        self.json_data = json.loads(js)
        self.camera_list = camera_list
        self.coordinates_len = coordinates_len
        self.mode = mode
        self.ccl = None
        self.ccr = None
        self.r = None
        self.t = None
        self.f = None
        self.calibrate()
        
        self.parallelized_img_r = None
        self.world_coordinates = {}
        self.prepare_image(bg_num)
        self.height = self.imgs.shape[2]
        self.width = self.imgs.shape[3]
        print("\tDone.")
        if GPU:
            self.imgs_gray_cp = cp.asarray(self.imgs_gray)
        else:
            self.imgs_gray_cp = None
        if (window_size%2 == 0):
            self.window_size = window_size
        else:
            self.window_size = window_size + 1
        self.diff = {}
        self.diff_blur = {}
        self.epipolor_line = None
        self.epipolor_line_list = None
        self.cpu_rate = cpu_rate
    
    def prepare_image(self, bg_num):
        print("prepairing imgs...", end="")
        bg_imgs = []
        for i, img_path in enumerate(self.json_data["img_paths"][bg_num]):
            bg_img = cv2.imread(img_path)
            bg_img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
            bg_imgs.append(bg_img_gray)
        
        imgs = []
        imgs_gray = []
        for i, pair in enumerate(self.json_data["img_paths"]):
            p = []
            p_gray = []
            for cam, e in enumerate(pair):
                img = cv2.imread(e)
                if cam not in self.camera_list:
                    p.append(np.zeros_like(img))
                    p_gray.append(np.zeros(img.shape[:2]))
                    continue
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                diff = (img_gray - bg_imgs[cam])
                img_gray = np.where((diff > 120), img_gray, 0)
                # img_gray = blur_bilateral(img_gray)
                # img_gray = min_max(img_gray)
                # img_gray = diff_mean(img_gray)
                # img_gray = cv2.equalizeHist(img_gray)
                # img_gray = zscore(img_gray)
                # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
                
                if (cam == 0 and i == 66):
                    cv2.imshow("bg", bg_imgs[cam])
                    cv2.imshow("diff", diff)
                    cv2.imshow("target", img_gray)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                p.append(img)
                p_gray.append(img_gray)
            
            imgs.append(p)
            imgs_gray.append(p_gray)
            print("\rprepairing imgs... %d / %d" %(i + 1, len(self.json_data["img_paths"])), end="")
        self.imgs = np.asarray(imgs)
        self.imgs_gray = np.asarray(imgs_gray, dtype=np.uint8)
     
    def calibrate(self, calibrate_img_num=0):
        print("Calibrating... ", end="")
        pairs = self.json_data["calibration_img_paths"].keys()
        camera_list = [str(n) for n in self.camera_list]
        found = False
        r_list = []
        t_list = []
        ar_list = []
        al_list = []
        for pair in pairs:
            pair_list = pair.split(",")
            if all(num in pair_list for num in camera_list):
                calibration_img_paths = self.json_data["calibration_img_paths"][pair][-1]
                calibration_img_path_l = calibration_img_paths[pair_list.index(camera_list[0])]
                calibration_img_path_r = calibration_img_paths[pair_list.index(camera_list[1])]
                ccl = CameraCalibration(coordinates_len=self.coordinates_len, img_path=calibration_img_path_l)
                ccr = CameraCalibration(coordinates_len=self.coordinates_len, img_path=calibration_img_path_r)
                ccl.load_params(self.json_path, self.camera_list[0], show=False)
                ccr.load_params(self.json_path, self.camera_list[1], show=False)
                al, rl, tl = ccl.calc_params()
                ar, rr, tr = ccr.calc_params()
                rl_inv = np.linalg.inv(rl)
                r = np.dot(rr, rl_inv)
                t = np.dot(r, tl) - tr
                r_list.append(r)
                t_list.append(t)
                ar_list.append(ar)
                al_list.append(al)
                
                self.ccl = ccl
                self.ccr = ccr
                found = True
        if not found:
            print("No calibration image exists.")
            exit(1)
        self.r = np.mean(np.asarray(r_list), axis=0)
        self.t = np.mean(np.asarray(t_list), axis=0)
        self.ar = np.mean(np.asarray(ar_list), axis=0)
        self.al = np.mean(np.asarray(al_list), axis=0)
        
        print("Done.")
        print("R =\n", self.r)
        print("-"*50)
        print("T =\n", self.t)
        print("-"*50)
        print("A_R =\n", self.ar)
        print("-"*50)
        print("A_L =\n", self.al)
        print("-"*50)
        
    def get_world_coordinates(self, img_num, blur_filter, segmantation):
        try:
            return self.world_coordinates[str(img_num)]
        except KeyError:
            self.calc_world_coordinates(img_num, blur_filter, segmantation)
            return self.world_coordinates[str(img_num)]
    
    def calc_world_coordinates(self, img_num, blur_filter, segmantation):
        diff = self.get_diff(img_num, blur_filter, segmantation)
        self.world_coordinates[str(img_num)] = np.zeros((self.height, self.width, 3))
        for x in range(0, self.width):
            for y in range(0, self.height):
                self.world_coordinates[str(img_num)][y, x, :] = [x, self.height-y, (diff[y, x])]
    
    def create_ply(self, img_num, blur_filter, segmentaion):
        pcd = o3d.geometry.PointCloud()
        world_coordinates = self.get_world_coordinates(img_num, blur_filter, segmentaion)
        print("Creating Point Cloud... ", end="")
        for x in range(self.window_size//2, self.width - self.window_size//2):
            for y in range(self.window_size//2, self.height - self.window_size//2):
                if (self.diff[str(img_num)][y, x] < 2):
                    continue
                pcd.points.append(world_coordinates[y, x])
                bgr = self.imgs[img_num, 0, y, x] / 255
                pcd.colors.append([bgr[2], bgr[1], bgr[0]])
        print("Done.")
        print("Estimating normals... ", end="")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10)
        print("Done.")
        print("Creating mesh... ", end="")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)/2
        radius = 2*avg_dist
        radii = [radius, radius * 2]
        recMeshBPA = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        print("Done.")
        o3d.visualization.draw_geometries([recMeshBPA])
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        o3d.io.write_triangle_mesh(self.save_path + img_path[img_path.rfind("/"):-4] + "_"+ str(self.window_size) + ".ply",recMeshBPA)
    
    def show_3d(self, img_num, blur_filter=None, segmentation=None):
        world_coordinates = self.get_world_coordinates(img_num, blur_filter, segmentation)
        cnt = 0
        pcd = o3d.geometry.PointCloud()
        if (self.mode < 2 or self.mode == 4):
            rectified, _ = self.rectify(img_num)
            for x in range(self.window_size//2, self.width - self.window_size//2):
                for y in range(self.window_size//2, self.height - self.window_size//2):
                    if (self.mode == 4):
                        bgr = rectified[y, x] / 255
                        if (np.sum(bgr) == 0):
                            continue
                        pcd.points.append(world_coordinates[y, x])
                        pcd.colors.append([bgr[2], bgr[1], bgr[0]])
                    else:
                        bgr = self.imgs[img_num, self.camera_list[0], y, x] / 255
                        pcd.points.append(world_coordinates[y, x])
                        pcd.colors.append([bgr[2], bgr[1], bgr[0]])
        else:
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    pcd.points.append(world_coordinates[y, x])
                    bgr = self.imgs[img_num, self.camera_list[0], y, x] / 255
                    pcd.colors.append([bgr[2], bgr[1], bgr[0]])
            
        print(cnt)
        o3d.visualization.draw_geometries(
            [pcd],
            width=self.width,
            height=self.height,
            point_show_normal = False,
            )
    
    def get_move(self):
        rl_inv = np.linalg.inv(self.rl)
        self.r = np.dot(self.rr, rl_inv)
        self.t = self.tl - np.dot(self.r, self.tr)
        print(self.r)
        print(self.t)
        print("-"*50)
        return self.r, self.t
    
    def get_f(self):
        if (self.f is None):
            return self.calc_f_from_params()
        else:
            return self.f
    
    def calc_f_from_params(self):
        al_inv = np.linalg.inv(self.al)
        ar_inv = np.linalg.inv(self.ar)
        ar_inv_t = ar_inv.T.copy()
        tx = np.zeros((3, 3), dtype=np.float64)
        tx[0, 1] = self.t[2] * -1
        tx[0, 2] = self.t[1]
        tx[1, 0] = self.t[2]
        tx[1, 2] = self.t[0] * -1
        tx[2, 0] = self.t[1] * -1
        tx[2, 1] = self.t[0]
        
        self.f = np.dot(np.dot(np.dot(ar_inv_t, tx), self.r), al_inv)
        self.f.astype(np.int16)
        return self.f
    
    def get_diff(self, img_num, blur_filter, segmentation):
        try:
            diff = self.diff[str(img_num)]
        except KeyError:
            if self.cpu_rate is None and (self.mode <= 3):
                diff = self.calc_diff_r3d(img_num)
            elif (self.cpu_rate is None) and (4 <= self.mode):
                diff = self.calc_diff_cv2(img_num)
            else:
                diff = self.calc_diff_multi(img_num)
        except Exception:
            print("get_deff()")
            exit(1)
        
        diff = diff.astype(np.float32)
        if (self.mode <= 2):
            half = self.window_size//2
            d = diff[half:-half, half:-half]
            diff = np.zeros_like(diff)
            diff[half:-half, half:-half] = d
            
        elif(self.mode == 3):
            b = np.arange(self.width-2, 0, -1, dtype=np.float32)
            b = np.tile(b, self.height-2).reshape(self.height-2, self.width-2)
            d = diff[1:-1, 1:-1]
            d = np.divide(d, b)
            d = np.where((d > 1), 0, d)
            d *= self.width // 2
            diff = np.zeros_like(diff)
            diff[1:-1, 1:-1] = d
            
        print("max: ", np.max(diff))
        print("min: ", np.min(diff))
        print("min abs", np.min(np.abs(diff)))
        if segmentation is not None:
            diff = self.segmentation(img_num, diff, segmentation)
        
        if ((blur_filter is not None) and self.mode < 2):
            d = diff[half:-half, half:-half]
            if (type(blur_filter) == list):
                for flt in blur_filter:
                    d = flt(d)
            else:
                d = blur_filter(d)
            diff[half:-half, half:-half] = d
        elif (blur_filter is not None and self.mode == 3):
            d = diff[1:-1, 1:-1]
            diff[1:-1, 1:-1] = blur_bilateral(d)
        
        plt.imshow(diff, cmap="gray")
        plt.show()
        return diff
    
    def calc_diff(self, img_num):
        ytensor = self.get_epipolor_line()
        
        size = self.width
        diff = np.zeros((self.height, self.width), dtype=np.float64)
        
        for x in range(0, size, 1):
            for y in range(0, self.height, 1):
                diff[y, x] = self.sad_minimise_2d(x, y, ytensor[x, y, :], img_num)
        print()
        diff = np.where((diff > self.width/2), self.width//2, diff)
        print("max: ", np.max(diff))
        print("min: ", np.min(diff))
        print("min abs", np.min(np.abs(diff)))
        self.diff[str(img_num)] = diff
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        np.save(self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "_diff.npy", diff)
        plt.imshow(diff, cmap="gray")
        cv2.imwrite(self.save_path + img_path[img_path.rfind("/"):-4] + "_"+ str(self.window_size) + "_diff_plt.png", diff)
        plt.show()
    
    def calc_diff_cv2(self, img_num):
        img1_rectified, img2_rectified = self.rectify(img_num)
        window_size = self.window_size - 1
        stereo = cv2.StereoSGBM_create(minDisparity = -16,
            numDisparities = 128,
            blockSize = 16,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            preFilterCap = 0,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 2,
            mode = 1
        )
        disp = stereo.compute(img1_rectified, img2_rectified)
        disp = cv2.normalize(disp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        self.diff[str(img_num)] = disp
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        np.save(self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "mode" + str(self.mode) + "_diff.npy", disp)
        plt.imshow(disp, cmap="gray")
        cv2.imwrite(self.save_path + img_path[img_path.rfind("/"):-4] + "_"+ str(self.window_size) + "mode" + str(self.mode) + "_diff_plt.png", disp)
        return disp
    
    def calc_diff_multi(self, img_num):
        ytensor = self.get_epipolor_line()
        
        diff = np.zeros((self.height, self.width), dtype=np.float64)
        
        args_list = []
        for x in range(0, self.width, 1):
            for y in range(0, self.height, 1):
                args_list.append((x, y, ytensor[x, y, :], img_num))
        start_time = datetime.now()
        print("start time: %s" %start_time)
        with Pool(int(multiprocessing.cpu_count() * self.cpu_rate)) as pool:
            result = pool.map(self.sad_minimise_multi, args_list)
        print()
        end_time = datetime.now()
        print("total time: %s" %(end_time - start_time))
        diff = np.reshape(result, (self.height, self.width), order="F")
        
        diff = np.where((diff > self.width/2), self.width//2, diff)
        print("max: ", np.max(diff))
        print("min: ", np.min(diff))
        print("min abs", np.min(np.abs(diff)))
        self.diff[str(img_num)] = diff
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        np.save(self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "_diff.npy", diff)
        plt.imshow(diff, cmap="gray")
        cv2.imwrite(self.save_path + img_path[img_path.rfind("/"):-4] + "_"+ str(self.window_size) + "_diff_plt.png", diff)
        plt.show()
        return diff
    
    def calc_diff_r3d(self, img_num):
        img1 = self.imgs_gray[img_num, self.camera_list[0]].tolist()
        img2 = self.imgs_gray[img_num, self.camera_list[1]].tolist()
        start_time = datetime.now()
        num_disparity = self.width // 3
        min_disparity = self.width // 80 * -1
        print("start time: %s" %start_time)
        if (0 <= self.mode and self.mode <= 2):
            epipolor = self.get_epipolor_line_list()
            diff = r3d.calc_diff(img1, img2, epipolor, self.window_size, self.mode, num_disparity, min_disparity)
        elif(3 <= self.mode):
            img1_rectified, img2_rectified = self.rectify(img_num)
            if not os.path.exists("./data/TEMP"):
                os.mkdir("./data/TEMP")
            img1_tmp_path = "./data/TEMP/img1.png"
            img2_tmp_path = "./data/TEMP/img2.png"
            cv2.imwrite(img1_tmp_path, img1_rectified)
            cv2.imwrite(img2_tmp_path, img2_rectified)
            diff = r3d.calc_diff_cv2(img1_tmp_path, img2_tmp_path, self.window_size, self.mode)
            os.remove(img1_tmp_path)
            os.remove(img2_tmp_path)
            
        end_time = datetime.now()
        print("Done.")
        print("total time: %s" %(end_time - start_time))
        diff = np.asarray(diff, dtype=np.int32)
        if (0 <= self.mode and self.mode <= 2):
            diff = np.where((diff >= num_disparity), min_disparity, diff)
            diff = np.where((diff < min_disparity), min_disparity, diff)
        if (np.min(diff) < 0):
            diff -= np.min(diff)
        
        print("max: ", np.max(diff))
        print("min: ", np.min(diff))
        print("min abs", np.min(np.abs(diff)))
        self.diff[str(img_num)] = diff
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        np.save(self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "mode" + str(self.mode) + "_diff.npy", diff)
        cv2.imwrite(self.save_path + img_path[img_path.rfind("/"):-4] + "_"+ str(self.window_size) + "mode" + str(self.mode) + "_diff_plt.png", diff)
        return diff
    
    def rectify(self, img_num):
        Rl, Rr, Pl, Pr, Q, validPixROIl, validPixROIr = cv2.stereoRectify(self.al, self.ccl.dist, self.ar, self.ccr.dist, (self.width, self.height), self.r, self.t, alpha=1, flags=0)
        mapXLeft, mapYLeft = cv2.initUndistortRectifyMap(self.al, self.ccl.dist, Rl, Pl, (self.width, self.height), cv2.CV_32FC1)
        mapXRight, mapYRight = cv2.initUndistortRectifyMap(self.ar, self.ccr.dist, Rr, Pr, (self.width, self.height), cv2.CV_32FC1)
        img_rectified_l = cv2.remap(self.imgs[img_num, self.camera_list[0]], mapXLeft, mapYLeft, cv2.INTER_LINEAR)
        img_rectified_r = cv2.remap(self.imgs[img_num, self.camera_list[1]], mapXRight, mapYRight, cv2.INTER_LINEAR)
        return img_rectified_l, img_rectified_r
        
    def sad_minimise_multi(self, args):
        x, y, line, img_num = args
        if (y == 0):
            print("\r%d / %d" %(x, self.width), end="")
        if GPU:
            d_min = self.sad_minimise_2d_cp(x, y, line, img_num)
        else:
            d_min = self.sad_minimise_2d(x, y, line, img_num)
        return d_min
    
    def calc_sad(self, window1, window2):
        try:
            dif = np.abs(window2 - window1)
            sad = np.sum(dif)
            return sad
        except ValueError:
            return 3**3 * 10**4
    
    def calc_sad_tensor(self, window1, window2):
        dif = np.abs(window2 - window1)
        sad = np.sum(dif, axis=(1))
        return sad
    
    def calc_sad_tensor_2d(self, window1, window2):
        dif = np.abs(window2 - window1)
        sad = np.sum(dif, axis=(1, 2))
        return sad
    
    def calc_sad_tensor_2d_cp(self, window1, window2):
        dif = cp.abs(window2 - window1)
        sad = cp.sum(dif, axis=(1, 2))
        return sad
    
    def calc_sad_weight_tensor_2d(self, window1, window2, line, i):
        diff = np.abs(window2 - window1)
        ilist = np.full(window1.shape[0]*(self.window_size**2), i).reshape(window1.shape[0], self.window_size, self.window_size)
        std = np.std(line)
        if (std > 0):
            w = np.exp(-np.square(ilist - window1)/(2 * (std ** 2)))
        else:
            w = 1
        sad = np.dot(w, diff)
        sad = np.sum(diff, axis=(1, 2))
        return sad
    
    def calc_sad_weight_tensor_2d_cp(self, window1, window2, line, i):
        diff = cp.abs(window2 - window1)
        ilist = cp.full(window1.shape[0]*(self.window_size**2), i).reshape(window1.shape[0], self.window_size, self.window_size)
        std = cp.std(line)
        if (std > 0):
            w = cp.exp(-cp.square(ilist - window1)/(2 * (std ** 2)))
            diff = cp.dot(w, diff)
        sad = cp.sum(diff, axis=(1, 2))
        return sad
    
    def sad_minimise(self, x, y, line, img_num):
        d_min = float("inf")
        window1 = self.imgs_gray[img_num, 0, y, x:(x+self.window_size)]
        if (len(window1.shape) < 1 or (window1.shape[0] < self.window_size)):
            return d_min
        window1_list = []
        window2_list = []
        d_list = []
        for d in range((x - self.width+1), x, 1):
            y2 = line[x-d]
            if (y2 < 0):
                continue
            window2 = self.imgs_gray[img_num, 0, y2, x:(x+self.window_size)]
            if ((len(window1.shape) < 1) or (window2.shape[0] < self.window_size)):
                continue
            window1_list.append(window1)
            window2_list.append(window2)
            d_list.append(d)
        if (len(d_list) == 0):
            return d_min
        else:
            window1_list = np.asarray(window1_list)
            window2_list = np.asarray(window2_list)
            if (len(window2_list.shape) < 1):
                return d_min
            sad_list = self.calc_sad_tensor(window1_list, window2_list)
            d_min = d_list[np.argmin(sad_list)]
            return d_min
    
    def sad_minimise_2d(self, x, y, line, img_num):
        d_min = float("inf")
        if (x < self.window_size//2 or y < self.window_size//2
            or x > self.width + self.window_size//2
            or y > self.height + self.window_size//2):
            return d_min
        window1 = self.imgs_gray[img_num, 0, (y-self.window_size//2):(y+self.window_size//2), (x-self.window_size//2):(x+self.window_size//2)]
        
        if (window1.shape[0] < self.window_size or window1.shape[1] < self.window_size):
            return d_min
        window1_list = []
        window2_list = []
        d_list = []
        for x2 in range(0, self.width, 1):
            d = abs(x2 - x)
            y2 = line[x2]
            if ((y2 < self.window_size//2) or (y2 > self.height + self.window_size//2)):
                continue
            window2 = self.imgs_gray[img_num, 1, (y2-self.window_size//2):(y2+self.window_size//2), (x-self.window_size//2):(x+self.window_size//2)]
            if (window2.shape[0] < self.window_size or window2.shape[1] < self.window_size):
                continue
            window1_list.append(window1)
            window2_list.append(window2)
            d_list.append(d)
        
        if (len(d_list) == 0):
            return d_min
        
        else:
            window2_list = np.asarray(window2_list)
            window1_list = np.asarray(window1_list)
            sad_list = self.calc_sad_weight_tensor_2d(window1_list, window2_list, line, self.imgs_gray[img_num, 0, y, x])
            
            d_min = d_list[np.argmin(sad_list)]
            return d_min

    def sad_minimise_2d_cp(self, x, y, line, img_num):
        d_min = float("inf")
        if (x < self.window_size//2 or y < self.window_size//2
            or x > self.width + self.window_size//2
            or y > self.height + self.window_size//2):
            return d_min
        window1 = self.imgs_gray_cp[img_num, 0, (y-self.window_size//2):(y+self.window_size//2), (x-self.window_size//2):(x+self.window_size//2)]
        
        if (window1.shape[0] < self.window_size or window1.shape[1] < self.window_size):
            return d_min
        window1_list = []
        window2_list = []
        d_list = []
        for x2 in range(0, self.width, 1):
            d = abs(x2 - x)
            y2 = line[x2]
            if ((y2 < self.window_size//2) or (y2 > self.height + self.window_size//2)):
                continue
            window2 = self.imgs_gray_cp[img_num, 1, (y2-self.window_size//2):(y2+self.window_size//2), (x-self.window_size//2):(x+self.window_size//2)]
            if (window2.shape[0] < self.window_size or window2.shape[1] < self.window_size):
                continue
            window1_list.append(window1)
            window2_list.append(window2)
            d_list.append(d)
        
        if (len(d_list) == 0):
            return d_min
        
        else:
            window1_list = cp.asarray(window1_list)
            window2_list = cp.asarray(window2_list)
            sad_list = self.calc_sad_tensor_2d_cp(window1_list, window2_list)
            sad_list = cp.asnumpy(sad_list)
            d_min = d_list[np.argmin(sad_list)]
            return d_min
             
    def show_fundamental(self, img_num):
        ytensor = self.get_epipolor_line()
        for x in range(0, self.width, 60):
            for y in range(0, self.height, 10):
                xl = np.arange(0, self.width)
                yl = ytensor[x, y, :]
                show_right = self.imgs[img_num, 1].copy()
                show_left = self.imgs[img_num, 0].copy()
                cv2.drawMarker(show_left, (x, y), (255, 255, 150), markerSize=20, markerType=cv2.MARKER_STAR, thickness=1)
                show_right[yl, xl] = (255, 255, 100)
                
                show_img = cv2.hconcat([show_left, show_right])
                cv2.imshow("epipolar line", show_img)
                cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calc_epipolor_line(self):
        f = self.get_f()
        print("Calculating epipolor line... ", end="")
        points_xy = []
        for x in range(0, self.width):
            for y in range(self.height - 1, -1, -1):
                point_xy = [x, y, 1]
                points_xy.append(point_xy)
        points_xy = np.asarray(points_xy, dtype=np.int16)
        lines_factor = np.tensordot(f, points_xy, axes=(1, 1)).T
        lines_factor = np.reshape(lines_factor, (self.width, self.height, 3))
        
        xyzlist = np.zeros((self.width, self.height, self.width))
        xyzlist[:, :, :] = np.arange(0, self.width, 1, dtype=np.int16)
        a = lines_factor[:, :, 0]
        b = lines_factor[:, :, 1]
        c = lines_factor[:, :, 2]
        a = np.reshape(a, (self.width, self.height, 1))
        b = np.reshape(b, (self.width, self.height, 1))
        c = np.reshape(c, (self.width, self.height, 1))
        ytensor = self.height +  (xyzlist*a + c)/b
        ytensor = np.where((ytensor < 0), -1, ytensor)
        ytensor = np.where((ytensor > self.height), -1, ytensor)
        ytensor = ytensor.astype(np.int16)
        self.epipolor_line = ytensor
        print("Done.")
        return self.epipolor_line
    
    def get_epipolor_line(self):
        if self.epipolor_line is None:
            self.calc_epipolor_line()
        
        return self.epipolor_line

    def get_epipolor_line_list(self):
        pickle_path = self.json_path[:-12] + "/" + str(self.camera_list) + ".epip"
        if (self.epipolor_line_list is None and (not os.path.exists(pickle_path))):
            self.get_epipolor_line()
            print("Prepareing epipolor list... ", end="")
            self.epipolor_line_list = self.epipolor_line.tolist()
            self.epipolor_line = None
            with open(pickle_path, "wb") as f:
                pickle.dump(self.epipolor_line_list, f)
            print("Done.")
        elif (self.epipolor_line_list is None and os.path.exists(pickle_path)):
            print("Loading epipolor list... ", end="")
            with open(pickle_path, "rb") as f:
                self.epipolor_line_list = pickle.load(f)
            print("Done.")
        
        return self.epipolor_line_list
    
    def load_diff(self, img_num=None):
        if img_num is None:
            for i in range(len(self.json_data["img_paths"])):
                img_path = self.json_data["img_paths"][i][self.camera_list[0]]
                diff_path = self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "mode" + str(self.mode) + "_diff.npy"
                if os.path.exists(diff_path):
                    self.diff[str(img_num)] = np.load(diff_path)
                else:
                    print("Warning: Cannot load diff [" + diff_path + "]")
            return
        self.get_f()
        img_path = self.json_data["img_paths"][img_num][self.camera_list[0]]
        diff_path = self.save_path + img_path[img_path.rfind("/"):-4] + "_" + str(self.window_size) + "mode" + str(self.mode) + "_diff.npy"
        if os.path.exists(diff_path):
            self.diff[str(img_num)] = np.load(diff_path)
        else:
            print("No diff data exist.")
            print(diff_path)
            exit(1)
    
    def segmentation(self, img_num, diff, segmantation_func):
        if self.mode == 4:
            rectified, _ = self.rectify(img_num)
            converted = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV_FULL)
        else:
            converted = cv2.cvtColor(self.imgs[img_num, self.camera_list[0]], cv2.COLOR_BGR2HSV_FULL)
        seg = segmantation_func(converted)
        labels = seg.getLabels()
        
        nseg = seg.getNumberOfSuperpixels()
        print("nseg: ", nseg)
        new_diff = np.zeros((self.height, self.width), dtype=np.float32)
        for m in range(0, nseg):
            diff_mean = cv2.mean(diff[labels == m])[0]
            new_diff[labels == m] = diff_mean
        return new_diff

    def seed_segmentation(self, converted):
        num_iterations = 5
        prior = 2
        double_step = False
        num_superpixels = 150
        num_levels = 4
        num_histogram_bins = 5

        seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width, self.height, 3, num_superpixels,
                num_levels, prior, num_histogram_bins, double_step)
        seeds.iterate(converted, num_iterations)
        return seeds
    
    def slc_segmentation(self, converted):
        region_size = 70
        ruler = 0.001
        min_element_size = 140
        num_iterations = 6
        
        slc = cv2.ximgproc.createSuperpixelLSC(
            converted, region_size, float(ruler))
        slc.iterate(num_iterations)
        slc.enforceLabelConnectivity(min_element_size)
        return slc


# Mathmatical functions
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore

def diff_mean(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    d_mean = x - xmean
    return d_mean

def blur_bilateral(img):
    b = cv2.bilateralFilter(img.astype(np.uint8), 10, 10, 10)
    return b

def blur(img):
    b = cv2.blur(img.astype(np.uint8), (3, 3))
    return b

def blur_median(img):
    b = cv2.medianBlur(img.astype(np.uint8), 49).astype(np.int32)
    return b

def blur_gaussian(img):
    kernel_size = 3
    b = cv2.GaussianBlur(img.astype(np.uint8), (kernel_size, kernel_size), 0).astype(np.int32)
    return b

def blur_convolve(img):
    height = img.shape[0]
    width = img.shape[1]
    window = 29
    w = np.ones(window) / window
    b = img.flatten()
    b = np.convolve(b, w, mode="same")
    b = b.reshape(height, width).T.flatten()
    b = np.convolve(b, w, mode="same")
    b = b.reshape(width, height).T
    return b
