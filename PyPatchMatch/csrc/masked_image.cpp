#include "masked_image.h"
#include <algorithm>
#include <iostream>

const cv::Size MaskedImage::kDownsampleKernelSize = cv::Size(6, 6);
const int MaskedImage::kDownsampleKernel[6] = {1, 5, 10, 10, 5, 1};

bool MaskedImage::contains_mask(int y, int x, int patch_size) const {
    auto mask_size = size();
    for (int dy = -patch_size; dy <= patch_size; ++dy) {
        for (int dx = -patch_size; dx <= patch_size; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < mask_size.height && xx >= 0 && xx < mask_size.width) {
                if (is_masked(yy, xx) && !is_globally_masked(yy, xx)) return true;
            }
        }
    }
    return false;
}

MaskedImage MaskedImage::downsample() const {
    const auto &kernel_size = MaskedImage::kDownsampleKernelSize;
    const auto &kernel = MaskedImage::kDownsampleKernel;

    const auto size = this->size();
    const auto new_size = cv::Size(size.width / 2, size.height / 2);

    auto ret = MaskedImage(new_size.width, new_size.height);
    if (!m_global_mask.empty()) ret.init_global_mask_mat();
    for (int y = 0; y < size.height - 1; y += 2) {
        for (int x = 0; x < size.width - 1; x += 2) {
            int r = 0, g = 0, b = 0, ksum = 0;
            bool is_gmasked = true;

            for (int dy = -kernel_size.height / 2 + 1; dy <= kernel_size.height / 2; ++dy) {
                for (int dx = -kernel_size.width / 2 + 1; dx <= kernel_size.width / 2; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < size.height && xx >= 0 && xx < size.width) {
                        if (!is_globally_masked(yy, xx)) {
                            is_gmasked = false;
                        }
                        if (!is_masked(yy, xx)) {
                            auto source_ptr = get_image(yy, xx);
                            int k = kernel[kernel_size.height / 2 - 1 + dy] * kernel[kernel_size.width / 2 - 1 + dx];
                            r += source_ptr[0] * k, g += source_ptr[1] * k, b += source_ptr[2] * k;
                            ksum += k;
                        }
                    }
                }
            }

            if (ksum > 0) r /= ksum, g /= ksum, b /= ksum;

            if (!m_global_mask.empty()) {
                ret.set_global_mask(y / 2, x / 2, is_gmasked);
            }
            if (ksum > 0) {
                auto target_ptr = ret.get_mutable_image(y / 2, x / 2);
                target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
                ret.set_mask(y / 2, x / 2, 0);
            } else {
                ret.set_mask(y / 2, x / 2, 1);
            }
        }
    }

    return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h) const {
    const auto size = this->size();
    auto ret = MaskedImage(new_w, new_h);
    if (!m_global_mask.empty()) ret.init_global_mask_mat();
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int yy = y * size.height / new_h;
            int xx = x * size.width / new_w;

            if (is_globally_masked(yy, xx)) {
                ret.set_global_mask(y, x, 1);
                ret.set_mask(y, x, 1);
            } else {
                if (!m_global_mask.empty()) ret.set_global_mask(y, x, 0);

                if (is_masked(yy, xx)) {
                    ret.set_mask(y, x, 1);
                } else {
                    auto source_ptr = get_image(yy, xx);
                    auto target_ptr = ret.get_mutable_image(y, x);
                    for (int c = 0; c < 3; ++c)
                        target_ptr[c] = source_ptr[c];
                    ret.set_mask(y, x, 0);
                }
            }
        }
    }

    return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h, const cv::Mat &new_global_mask) const {
    auto ret = upsample(new_w, new_h);
    ret.set_global_mask_mat(new_global_mask);
    return ret;
}

void MaskedImage::compute_image_gradients() {
    if (m_image_grad_computed) {
        return;
    }

    const auto size = m_image.size();
    m_image_grady = cv::Mat(size, CV_8UC3);
    m_image_gradx = cv::Mat(size, CV_8UC3);
    m_image_grady = cv::Scalar::all(0);
    m_image_gradx = cv::Scalar::all(0);

    for (int i = 1; i < size.height - 1; ++i) {
        const auto *ptr = m_image.ptr<unsigned char>(i, 0);
        const auto *ptry1 = m_image.ptr<unsigned char>(i + 1, 0);
        const auto *ptry2 = m_image.ptr<unsigned char>(i - 1, 0);
        const auto *ptrx1 = m_image.ptr<unsigned char>(i, 0) + 3;
        const auto *ptrx2 = m_image.ptr<unsigned char>(i, 0) - 3;
        auto *mptry = m_image_grady.ptr<unsigned char>(i, 0);
        auto *mptrx = m_image_gradx.ptr<unsigned char>(i, 0);
        for (int j = 3; j < size.width * 3 - 3; ++j) {
            mptry[j] = (ptry1[j] / 2 - ptry2[j] / 2) + 128;
            mptrx[j] = (ptrx1[j] / 2 - ptrx2[j] / 2) + 128;
        }
    }

    m_image_grad_computed = true;
}

void MaskedImage::compute_image_gradients() const {
    const_cast<MaskedImage *>(this)->compute_image_gradients();
}

