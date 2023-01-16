# please run setup.py at "./cpp" before run this program.

import stereo_vision as st

def main():
    json_path = "***/params.json"
    using_camera = [0, 1]
    img_num = 0
    cpu_rate = None
    window_size = 4
    mode = 1
    bg_num = 42
    sv = st.StereoVision(json_path, camera_list=using_camera, bg_num=bg_num, coordinates_len=18,
                         window_size=window_size, cpu_rate=cpu_rate, mode=mode)
    # sv.load_diff(img_num)
    sv.show_3d(img_num, blur_filter=[st.blur_gaussian, st.blur_median, st.blur_convolve], segmentation=None)
if __name__ == "__main__":
    main()