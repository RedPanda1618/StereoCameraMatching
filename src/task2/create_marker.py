import cv2
from cv2 import aruco
import os

dir_mark = "./data/markers"

num_mark = 18
size_mark = 500

dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)

for count in range(num_mark) :
    id_mark = count
    img_mark = aruco.drawMarker(dict_aruco, id_mark, size_mark)
    img_name_mark = "mark_id_" + str(format(count, '02')) + "png"
    path_mark = os.path.join(dir_mark, img_name_mark)
    print(path_mark)
    cv2.imwrite(path_mark, img_mark)