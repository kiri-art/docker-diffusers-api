#pragma once

#include <opencv2/core.hpp>
#include "masked_image.h"

class PatchDistanceMetric {
public:
    PatchDistanceMetric(int patch_size) : m_patch_size(patch_size) {}
    virtual ~PatchDistanceMetric() = default;

    inline int patch_size() const { return m_patch_size; }
    virtual int operator()(const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const = 0;
    static const int kDistanceScale;

protected:
    int m_patch_size;
};

class NearestNeighborField {
public:
    NearestNeighborField() : m_source(), m_target(), m_field(), m_distance_metric(nullptr) {
        // pass
    }
    NearestNeighborField(const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric, int max_retry = 20)
        : m_source(source), m_target(target), m_distance_metric(metric) {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _randomize_field(max_retry);
    }
    NearestNeighborField(const MaskedImage &source, const MaskedImage &target, const PatchDistanceMetric *metric, const NearestNeighborField &other, int max_retry = 20)
            : m_source(source), m_target(target), m_distance_metric(metric) {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _initialize_field_from(other, max_retry);
    }

    const MaskedImage &source() const {
        return m_source;
    }
    const MaskedImage &target() const {
        return m_target;
    }
    inline cv::Size source_size() const {
        return m_source.size();
    }
    inline cv::Size target_size() const {
        return m_target.size();
    }
    inline void set_source(const MaskedImage &source) {
        m_source = source;
    }
    inline void set_target(const MaskedImage &target) {
        m_target = target;
    }

    inline int *mutable_ptr(int y, int x) {
        return m_field.ptr<int>(y, x);
    }
    inline const int *ptr(int y, int x) const {
        return m_field.ptr<int>(y, x);
    }

    inline int at(int y, int x, int c) const {
        return m_field.ptr<int>(y, x)[c];
    }
    inline int &at(int y, int x, int c) {
        return m_field.ptr<int>(y, x)[c];
    }
    inline void set_identity(int y, int x) {
        auto ptr = mutable_ptr(y, x);
        ptr[0] = y, ptr[1] = x, ptr[2] = 0;
    }

    void minimize(int nr_pass);

private:
    inline int _distance(int source_y, int source_x, int target_y, int target_x) {
        return (*m_distance_metric)(m_source, source_y, source_x, m_target, target_y, target_x);
    }

    void _randomize_field(int max_retry = 20, bool reset = true);
    void _initialize_field_from(const NearestNeighborField &other, int max_retry);
    void _minimize_link(int y, int x, int direction);

    MaskedImage m_source;
    MaskedImage m_target;
    cv::Mat m_field;  // { y_target, x_target, distance_scaled }
    const PatchDistanceMetric *m_distance_metric;
};


class PatchSSDDistanceMetric : public PatchDistanceMetric {
public:
    using PatchDistanceMetric::PatchDistanceMetric;
    virtual int operator ()(const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const;
    static const int kSSDScale;
};

class DebugPatchSSDDistanceMetric : public PatchDistanceMetric {
public:
    DebugPatchSSDDistanceMetric(int patch_size, int width, int height) : PatchDistanceMetric(patch_size), m_width(width), m_height(height) {}
    virtual int operator ()(const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const;
protected:
    int m_width, m_height;
};

class RegularityGuidedPatchDistanceMetricV1 : public PatchDistanceMetric {
public:
    RegularityGuidedPatchDistanceMetricV1(int patch_size, double dx1, double dy1, double dx2, double dy2, double weight)
        : PatchDistanceMetric(patch_size), m_dx1(dx1), m_dy1(dy1), m_dx2(dx2), m_dy2(dy2), m_weight(weight) {

        assert(m_dy1 == 0);
        assert(m_dx2 == 0);
        m_scale = sqrt(m_dx1 * m_dx1 + m_dy2 * m_dy2) / 4;
    }
    virtual int operator ()(const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const;

protected:
    double m_dx1, m_dy1, m_dx2, m_dy2;
    double m_scale, m_weight;
};

class RegularityGuidedPatchDistanceMetricV2 : public PatchDistanceMetric {
public:
    RegularityGuidedPatchDistanceMetricV2(int patch_size, cv::Mat ijmap, double weight)
        : PatchDistanceMetric(patch_size), m_ijmap(ijmap), m_weight(weight) {

    }
    virtual int operator ()(const MaskedImage &source, int source_y, int source_x, const MaskedImage &target, int target_y, int target_x) const;

protected:
    cv::Mat m_ijmap;
    double m_width, m_height, m_weight;
};

