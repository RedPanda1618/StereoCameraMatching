import multi_input

def main():
    mi = multi_input.MultiInput(2, width=1280, height=720)
    mi.capture()

if __name__ == "__main__":
    main()