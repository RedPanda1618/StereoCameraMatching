from multi_input import MultiInput

def main():
    # img_dir = "data/MultiCamera/20230106-122111"
    img_dir = "data/MultiCamera/20230110-145109"
    # img_dir = "../memo/23_0110_1312"

    # mi = MultiInput(5, width=1280, height=720)
    mi = MultiInput(5, width=854, height=480)
    mi.data_format(img_dir, 72, mode=0)

if "__main__" == __name__:
    main()