from distutils.core import setup, Extension
 
setup(name = 'reconstruction3d', version = '1.2.0', 
      ext_modules = [Extension('reconstruction3d', ['calc_img_diff.cpp'],
                              #  library_dirs=["/usr/local/include/opencv4"],
                              # libraries=["opencv_core", "opencv_imgcodecs", "opencv_highgui", "opencv_calib3d"],
                              # include_dirs=["/usr/include/opencv2", "/usr/include/eigen3"]
                               )]
      )