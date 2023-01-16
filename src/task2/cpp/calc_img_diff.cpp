#include "eigen3/Eigen/Dense"
#include <Python.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

int sad_minimise_2d(int &x, int &y, vector<int> &epipolor_line,
                    MatrixXf &img_1_mat, MatrixXf &img_2_mat, int &window_size,
                    int &width, int &heigth, int &mode, int &min_disparity,
                    int &num_disparity);

void slice_2d(MatrixXf &array, MatrixXf &new_array, int start_1d, int end_1d,
              int start_2d, int end_2d);

int calc_sad(MatrixXf &window1, MatrixXf &window2);

int calc_ssd(MatrixXf &window1, MatrixXf &window2);

void census_transform(MatrixXf &img, MatrixXi &census, int &width, int &height);

void calc_pixel_cost(vector<vector<vector<int>>> &epipolor_line,
                     MatrixXi &census1, MatrixXi &census2, int &census_width,
                     int &census_height, vector<vector<vector<int>>> &pix_cost);

void calc_pixel_cost_min(vector<vector<vector<int>>> &epipolor_line,
                         MatrixXi &census1, MatrixXi &census2,
                         int &census_width, int &census_height,
                         int &window_size, int &num_disparity,
                         int &min_disparity, vector<vector<int>> &pix_cost);
char calc_haming_dist(unsigned char val1, unsigned char val2);

void calc_cost(MatrixXi &window1, MatrixXi &window2, int &val);

static PyObject *calc_diff(PyObject *self, PyObject *args) {
    cout << "Calculate Module: [reconstruction3d]" << endl;
    int width, height, x, x2, y, window_size, d, mode, census_width,
        census_height, num_disparity, min_disparity;
    double px1, px2;
    PyObject *img_1_py, *img_2_py, *epipolor_line_py, *img1_col, *img2_col,
        *epipolor_col, *epipolor_row, *diff_row, *diff_py, *d_py;
    MatrixXf img1_mat, img2_mat;
    MatrixXi census1, census2;

    if(!PyArg_ParseTuple(args, "OOOiiii", &img_1_py, &img_2_py,
                         &epipolor_line_py, &window_size, &mode, &num_disparity,
                         &min_disparity)) {
        cout << "Arg Error" << endl;
        return NULL;
    }
    cout << "mode: " << mode << endl
         << "num disparity: " << num_disparity << endl
         << "min disparity: " << min_disparity << endl;
    // Check list
    if(!PyList_Check(img_1_py)) {
        cout << "List Error img_1";
        return NULL;
    }
    if(!PyList_Check(PyList_GetItem(img_1_py, 0))) {
        cout << "List Error img_1 2d";
        return NULL;
    }
    if(!PyList_Check(img_2_py)) {
        cout << "List Error img_2";
        return NULL;
    }
    if(!PyList_Check(PyList_GetItem(img_2_py, 0))) {
        cout << "List Error img_2 2d";
        return NULL;
    }
    if(!PyList_Check(epipolor_line_py)) {
        cout << "List Error Epipolor_line";
        return NULL;
    }
    if(!PyList_Check(PyList_GetItem(epipolor_line_py, 0))) {
        cout << "List Error epipolor_line 2d";
        return NULL;
    }
    if(!PyList_Check(PyList_GetItem(PyList_GetItem(epipolor_line_py, 0), 0))) {
        cout << "List Error epipolor_line 3d";
        return NULL;
    }

    height = PyList_Size(img_1_py);
    width = PyList_Size(PyList_GetItem(img_1_py, 0));
    img1_mat = MatrixXf::Zero(height, width);
    img2_mat = MatrixXf::Zero(height, width);

    vector<vector<int>> diff_vec(height, vector<int>(width)), epipolor;
    vector<vector<vector<int>>> epipolor_line(
        width, vector<vector<int>>(height, vector<int>(width, 0)));

    cout << "Preparing 1/3...";
    for(y = 0; y < height; y++) {
        // cout << "\rPreparing 1/2... " << y << " / " << height - 1;
        img1_col = PyList_GetItem(img_1_py, y);
        img2_col = PyList_GetItem(img_2_py, y);
        for(x = 0; x < width; x++) {
            px1 = PyFloat_AsDouble(PyList_GetItem(img1_col, x));
            px2 = PyFloat_AsDouble(PyList_GetItem(img2_col, x));
            img1_mat(y, x) = px1;
            img2_mat(y, x) = px2;
        }
    }
    cout << " Done." << endl;

    cout << "Preparing 2/3...";
    for(x = 0; x < width; x++) {
        if(x % 100 == 0) {
            cout << "\rPreparing 2/3... " << x << " / " << width - 1;
        }

        epipolor_col = PyList_GetItem(epipolor_line_py, x);
        for(y = 0; y < height; y++) {
            epipolor_row = PyList_GetItem(epipolor_col, y);
            for(x2 = 0; x2 < width; x2++) {
                epipolor_line[x][y][x2] =
                    PyLong_AsLong(PyList_GetItem(epipolor_row, x2));
            }
        }
    }

    cout << " Done." << endl;

    cout << "Preparing 3/3...";
    if(mode == 2) {
        census_transform(img1_mat, census1, width, height);
        census_transform(img2_mat, census2, width, height);
        census_width = width - 2;
        census_height = height - 2;
    }
    cout << " Done." << endl;

    cout << "Calculating...";
    switch(mode) {
    case 0:
    case 1:
        for(x = 0; x < width; x++) {
            epipolor = epipolor_line[x];
            if(x % 100 == 0) {
                cout << "\rCalculating... " << x << " / " << width - 1;
            }
            for(y = 0; y < height; y++) {
                d = sad_minimise_2d(x, y, epipolor[y], img1_mat, img2_mat,
                                    window_size, width, height, mode,
                                    min_disparity, num_disparity);
                diff_vec[y][x] = d;
            }
        }
        break;
    case 2:
        calc_pixel_cost_min(epipolor_line, census1, census2, census_width,
                            census_height, window_size, min_disparity,
                            num_disparity, diff_vec);
    default:
        break;
    }
    cout << "\rCalculating... Done." << endl;

    diff_py = PyList_New(height);
    cout << "Packing...";
    for(y = 0; y < height; y++) {
        // cout << "\rPacking... " << y << " / " << height - 1;
        diff_row = PyList_New(width);
        for(x = 0; x < width; x++) {
            d_py = PyLong_FromDouble(diff_vec[y][x]);
            PyList_SET_ITEM(diff_row, x, d_py);
        }
        PyList_SET_ITEM(diff_py, y, diff_row);
    }
    cout << " Done." << endl;

    return diff_py;
}

int sad_minimise_2d(int &x, int &y, vector<int> &epipolor, MatrixXf &img_1_mat,
                    MatrixXf &img_2_mat, int &window_size, int &width,
                    int &height, int &mode, int &min_disparity,
                    int &num_disparity) {
    if(img_1_mat(y, x) == 0) {
        return min_disparity;
    }
    int d_min = INT_MAX, sad, sad_min = INT_MAX, dis, x2, y2,
        half_window_size = window_size / 2;
    if((x < half_window_size) || (y < half_window_size) ||
       ((width - 1) < (x + half_window_size)) ||
       ((height - 1) < (y + half_window_size))) {
        return min_disparity;
    }
    MatrixXf window1;
    slice_2d(img_1_mat, window1, (y - half_window_size), (y + half_window_size),
             (x - half_window_size), (x + half_window_size));

    for(x2 = 0; x2 < width; x2++) {
        dis = x - x2;
        y2 = epipolor[x2];
        if((x2 < half_window_size) || (width - 1) < (x2 + half_window_size) ||
           (y2 < half_window_size) || (height - 1) < (y2 + half_window_size) ||
           (dis < min_disparity) || (dis > num_disparity)) {
            sad = INT_MAX;
        } else {
            MatrixXf window2;
            slice_2d(img_2_mat, window2, (y2 - half_window_size),
                     (y2 + half_window_size), (x2 - half_window_size),
                     (x2 + half_window_size));
            switch(mode) {
            case 0:
                sad = calc_sad(window1, window2);
                break;
            case 1:
                sad = calc_ssd(window1, window2);
                break;
            default:
                break;
            }
        }
        if(sad_min > sad) {
            sad_min = sad;
            d_min = dis;
        }
    }
    if(d_min == INT_MAX) {
        d_min = min_disparity;
    }
    return d_min;
}

int calc_sad(MatrixXf &window1, MatrixXf &window2) {
    MatrixXf diff;
    int sad;
    diff = window2 - window1;
    diff = diff.cwiseAbs();
    sad = diff.sum();
    return sad;
}

int calc_ssd(MatrixXf &window1, MatrixXf &window2) {
    MatrixXf diff;
    int sad;
    diff = window2 - window1;
    diff = diff.cwiseAbs2();
    sad = diff.sum();
    return sad;
}

void slice_2d(MatrixXf &array, MatrixXf &new_array, int start_1d, int end_1d,
              int start_2d, int end_2d) {
    int size_x, size_y;
    // MatrixXf new_array;
    size_x = end_2d - start_2d;
    size_y = end_1d - start_1d;
    if(end_1d <= start_1d) {
        cout << endl << "error: (end_1d <= start_1d)" << endl;
        exit(1);
    } else if(end_2d <= start_2d) {
        cout << endl << "error: (end_2d <= start_2d)" << endl;
        exit(1);
    } else if(start_1d < 0) {
        cout << endl << "error: (start_1d < 0)" << endl;
    } else if(start_2d < 0) {
        cout << endl << "error: (start_2d < 0)" << endl;
    } else if(end_1d < 0) {
        cout << endl << "error: (end_1d < 0)" << endl;
    } else if(end_2d < 0) {
        cout << endl << "error: (end_2d < 0)" << endl;
    }
    new_array = MatrixXf::Zero(size_y, size_x);
    int x, y;
    for(y = 0; y < size_y; y++) {
        for(x = 0; x < size_x; x++) {
            new_array(y, x) = array((start_1d + y), (start_2d + x));
        }
    }
}

void calc_pixel_cost(vector<vector<vector<int>>> &epipolor_line,
                     MatrixXi &census1, MatrixXi &census2, int &census_width,
                     int &census_height,
                     vector<vector<vector<int>>> &pix_cost) {
    unsigned char val1, val2, cost, min_cost;
    int d, min_d;
    for(int x = 0; x < census_width; x++) {
        cout << "\rCalculating pixel cost... " << x << " / "
             << census_width - 1;
        for(int y = 0; y < census_height; y++) {
            val1 = census1(y, x);
            min_d = census_width;
            min_cost = CHAR_MAX;
            for(int x2 = 0; x2 < census_width; x2++) {
                if(d >= 0 && epipolor_line[x][y][x2] > 0 &&
                   (epipolor_line[x][y][x2] < census_height - 1)) {
                    val2 = static_cast<unsigned char>(
                        census2(epipolor_line[x][y][x2]), x2);
                    cost = calc_haming_dist(val1, val2);
                    pix_cost[y][x][x2] = cost;
                }
            }
        }
    }
    cout << "Done." << endl;
    return;
}

void calc_pixel_cost_min(vector<vector<vector<int>>> &epipolor_line,
                         MatrixXi &census1, MatrixXi &census2,
                         int &census_width, int &census_height,
                         int &window_size, int &num_disparity,
                         int &min_disparity, vector<vector<int>> &pix_cost) {
    int y2, cost, min_cost, d, min_d, half_window_size = window_size / 2;
    MatrixXf census1f = census1.cast<float>();
    MatrixXf census2f = census2.cast<float>();
    for(int x = 0; x < census_width; x++) {
        cout << "\rCalculating pixel cost min... " << x << " / "
             << census_width - 1;
        if(x < half_window_size || census_width < x + half_window_size) {
            continue;
        }
        for(int y = 0; y < census_height; y++) {
            if(y < half_window_size || census_height < y + half_window_size) {
                continue;
            }
            min_d = census_width;
            min_cost = CHAR_MAX;
            MatrixXf window1;
            slice_2d(census1f, window1, (y - half_window_size),
                     (y + half_window_size), (x - half_window_size),
                     (x + half_window_size));
            MatrixXi window1i = window1.cast<int>();
            for(int x2 = 0; x2 < census_width; x2++) {
                d = x2 - x;
                y2 = epipolor_line[x][y][x2];
                if(x2 < half_window_size ||
                   census_width < x2 + half_window_size ||
                   y2 < half_window_size ||
                   census_height < y2 + half_window_size) {
                    continue;
                }
                if(min_disparity < d && d < num_disparity &&
                   y2 > half_window_size &&
                   (y2 < census_height - half_window_size)) {
                    MatrixXf window2;
                    slice_2d(census2f, window2, (y2 - half_window_size),
                             (y2 + half_window_size), (x2 - half_window_size),
                             (x2 + half_window_size));
                    MatrixXi window2i = window2.cast<int>();
                    calc_cost(window1i, window2i, cost);
                    if(cost < min_cost) {
                        min_cost = cost;
                        min_d = d;
                    }
                }
            }
            pix_cost[y][x] = min_d;
        }
    }
    cout << "Done." << endl;
}

void calc_cost(MatrixXi &window1, MatrixXi &window2, int &cost) {
    cost = 0;
    int rows, cols, i, j, d;
    rows = window1.rows();
    cols = window1.cols();

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            d = calc_haming_dist(static_cast<unsigned char>(window1(i, j)),
                                 static_cast<unsigned char>(window2(i, j)));
            cost += d;
        }
    }
}

void census_transform(MatrixXf &img, MatrixXi &census, int &width,
                      int &height) {
    census = MatrixXi::Zero(height - 2, width - 2);
    for(int x = 1; x < width - 1; x++) {
        for(int y = 1; y < height - 1; y++) {
            float center = img(y, x);
            unsigned char val = 0;
            for(int dx = -1; dx <= 1; dx++) {
                for(int dy = -1; dy <= 1; dy++) {
                    if(dx == 0 && dy == 0) {
                        continue;
                    }
                    float tmp = img(y + dy, x + dx);
                    val = (val + (tmp < center ? 0 : 1)) << 1;
                }
            }
            census(y - 1, x - 1) = val;
        }
    }
    return;
}

char calc_haming_dist(unsigned char val1, unsigned char val2) {
    unsigned char dist = 0;
    unsigned char d = val1 ^ val2;
    while(d) {
        d = d & (d - 1);
        dist++;
    }
    return dist;
}

// myModule definition(python's name)
static PyMethodDef methods[] = {
    {"calc_diff", calc_diff, METH_VARARGS, "calc differences from two images."},
    {NULL}};

// myModule definition struct
static struct PyModuleDef reconstruction3d = {
    PyModuleDef_HEAD_INIT, "reconstruction3d",
    "3D Reconstruction module width C++", -1, methods};

// Initializes myModule
PyMODINIT_FUNC PyInit_reconstruction3d(void) {
    return PyModule_Create(&reconstruction3d);
}