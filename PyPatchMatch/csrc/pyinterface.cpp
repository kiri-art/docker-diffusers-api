#include "pyinterface.h"
#include "inpaint.h"

static unsigned int PM_seed = 1212;
static bool PM_verbose = false;

int _dtype_py_to_cv(int dtype_py);
int _dtype_cv_to_py(int dtype_cv);
cv::Mat _py_to_cv2(PM_mat_t pymat);
PM_mat_t _cv2_to_py(cv::Mat cvmat);

void PM_set_random_seed(unsigned int seed) {
    PM_seed = seed;
}

void PM_set_verbose(int value) {
    PM_verbose = static_cast<bool>(value);
}

void PM_free_pymat(PM_mat_t pymat) {
    free(pymat.data_ptr);
}

PM_mat_t PM_inpaint(PM_mat_t source_py, PM_mat_t mask_py, int patch_size) {
    cv::Mat source = _py_to_cv2(source_py);
    cv::Mat mask = _py_to_cv2(mask_py);
    auto metric = PatchSSDDistanceMetric(patch_size);
    cv::Mat result = Inpainting(source, mask, &metric).run(PM_verbose, false, PM_seed);
    return _cv2_to_py(result);
}

PM_mat_t PM_inpaint_regularity(PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t ijmap_py, int patch_size, float guide_weight) {
    cv::Mat source = _py_to_cv2(source_py);
    cv::Mat mask = _py_to_cv2(mask_py);
    cv::Mat ijmap = _py_to_cv2(ijmap_py);

    auto metric = RegularityGuidedPatchDistanceMetricV2(patch_size, ijmap, guide_weight);
    cv::Mat result = Inpainting(source, mask, &metric).run(PM_verbose, false, PM_seed);
    return _cv2_to_py(result);
}

PM_mat_t PM_inpaint2(PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t global_mask_py, int patch_size) {
    cv::Mat source = _py_to_cv2(source_py);
    cv::Mat mask = _py_to_cv2(mask_py);
    cv::Mat global_mask = _py_to_cv2(global_mask_py);

    auto metric = PatchSSDDistanceMetric(patch_size);
    cv::Mat result = Inpainting(source, mask, global_mask, &metric).run(PM_verbose, false, PM_seed);
    return _cv2_to_py(result);
}

PM_mat_t PM_inpaint2_regularity(PM_mat_t source_py, PM_mat_t mask_py, PM_mat_t global_mask_py, PM_mat_t ijmap_py, int patch_size, float guide_weight) {
    cv::Mat source = _py_to_cv2(source_py);
    cv::Mat mask = _py_to_cv2(mask_py);
    cv::Mat global_mask = _py_to_cv2(global_mask_py);
    cv::Mat ijmap = _py_to_cv2(ijmap_py);

    auto metric = RegularityGuidedPatchDistanceMetricV2(patch_size, ijmap, guide_weight);
    cv::Mat result = Inpainting(source, mask, global_mask, &metric).run(PM_verbose, false, PM_seed);
    return _cv2_to_py(result);
}

int _dtype_py_to_cv(int dtype_py) {
    switch (dtype_py) {
        case PM_UINT8: return CV_8U;
        case PM_INT8: return CV_8S;
        case PM_UINT16: return CV_16U;
        case PM_INT16: return CV_16S;
        case PM_INT32: return CV_32S;
        case PM_FLOAT32: return CV_32F;
        case PM_FLOAT64: return CV_64F;
    }

    return CV_8U;
}

int _dtype_cv_to_py(int dtype_cv) {
    switch (dtype_cv) {
        case CV_8U: return PM_UINT8;
        case CV_8S: return PM_INT8;
        case CV_16U: return PM_UINT16;
        case CV_16S: return PM_INT16;
        case CV_32S: return PM_INT32;
        case CV_32F: return PM_FLOAT32;
        case CV_64F: return PM_FLOAT64;
    }

    return PM_UINT8;
}

cv::Mat _py_to_cv2(PM_mat_t pymat) {
    int dtype = _dtype_py_to_cv(pymat.dtype);
    dtype = CV_MAKETYPE(pymat.dtype, pymat.shape.channels);
    return cv::Mat(cv::Size(pymat.shape.width, pymat.shape.height), dtype, pymat.data_ptr).clone();
}

PM_mat_t _cv2_to_py(cv::Mat cvmat) {
    PM_shape_t shape = {cvmat.size().width, cvmat.size().height, cvmat.channels()};
    int dtype = _dtype_cv_to_py(cvmat.depth());
    size_t dsize = cvmat.total() * cvmat.elemSize();

    void *data_ptr = reinterpret_cast<void *>(malloc(dsize));
    memcpy(data_ptr, reinterpret_cast<void *>(cvmat.data), dsize);

    return PM_mat_t {data_ptr, shape, dtype};
}

