#include <algorithm>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "inpaint.h"

namespace {
    static std::vector<double> kDistance2Similarity;

    void init_kDistance2Similarity() {
        double base[11] = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0};
        int length = (PatchDistanceMetric::kDistanceScale + 1);
        kDistance2Similarity.resize(length);
        for (int i = 0; i < length; ++i) {
            double t = (double) i / length;
            int j = (int) (100 * t);
            int k = j + 1;
            double vj = (j < 11) ? base[j] : 0;
            double vk = (k < 11) ? base[k] : 0;
            kDistance2Similarity[i] = vj + (100 * t - j) * (vk - vj);
        }
    }


    inline void _weighted_copy(const MaskedImage &source, int ys, int xs, cv::Mat &target, int yt, int xt, double weight) {
        if (source.is_masked(ys, xs)) return;
        if (source.is_globally_masked(ys, xs)) return;

        auto source_ptr = source.get_image(ys, xs);
        auto target_ptr = target.ptr<double>(yt, xt);

#pragma unroll
        for (int c = 0; c < 3; ++c)
            target_ptr[c] += static_cast<double>(source_ptr[c]) * weight;
        target_ptr[3] += weight;
    }
}

/**
 * This algorithme uses a version proposed by Xavier Philippeau.
 */

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask), m_distance_metric(metric), m_pyramid(), m_source2target(), m_target2source() {
    _initialize_pyramid();
}

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, cv::Mat global_mask, const PatchDistanceMetric *metric)
    : m_initial(image, mask, global_mask), m_distance_metric(metric), m_pyramid(), m_source2target(), m_target2source() {
    _initialize_pyramid();
}

void Inpainting::_initialize_pyramid() {
    auto source = m_initial;
    m_pyramid.push_back(source);
    while (source.size().height > m_distance_metric->patch_size() && source.size().width > m_distance_metric->patch_size()) {
        source = source.downsample();
        m_pyramid.push_back(source);
    }

    if (kDistance2Similarity.size() == 0) {
        init_kDistance2Similarity();
    }
}

cv::Mat Inpainting::run(bool verbose, bool verbose_visualize, unsigned int random_seed) {
    srand(random_seed);
    const int nr_levels = m_pyramid.size();

    MaskedImage source, target;
    for (int level = nr_levels - 1; level >= 0; --level) {
        if (verbose) std::cerr << "Inpainting level: " << level << std::endl;

        source = m_pyramid[level];

        if (level == nr_levels - 1) {
            target = source.clone();
            target.clear_mask();
            m_source2target = NearestNeighborField(source, target, m_distance_metric);
            m_target2source = NearestNeighborField(target, source, m_distance_metric);
        } else {
            m_source2target = NearestNeighborField(source, target, m_distance_metric, m_source2target);
            m_target2source = NearestNeighborField(target, source, m_distance_metric, m_target2source);
        }

        if (verbose) std::cerr << "Initialization done." << std::endl;

        if (verbose_visualize) {
            auto visualize_size = m_initial.size();
            cv::Mat source_visualize(visualize_size, m_initial.image().type());
            cv::resize(source.image(), source_visualize, visualize_size);
            cv::imshow("Source", source_visualize);
            cv::Mat target_visualize(visualize_size, m_initial.image().type());
            cv::resize(target.image(), target_visualize, visualize_size);
            cv::imshow("Target", target_visualize);
            cv::waitKey(0);
        }

        target = _expectation_maximization(source, target, level, verbose);
    }

    return target.image();
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(MaskedImage source, MaskedImage target, int level, bool verbose) {
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = static_cast<int>(std::min(7, 1 + level));
    const int patch_size = m_distance_metric->patch_size();

    MaskedImage new_source, new_target;

    for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em) {
        if (iter_em != 0) {
            m_source2target.set_target(new_target);
            m_target2source.set_source(new_target);
            target = new_target;
        }

        if (verbose) std::cerr << "EM Iteration: " << iter_em << std::endl;

        auto size = source.size();
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                if (!source.contains_mask(i, j, patch_size)) {
                    m_source2target.set_identity(i, j);
                    m_target2source.set_identity(i, j);
                }
            }
        }
        if (verbose) std::cerr << "  NNF minimization started." << std::endl;
        m_source2target.minimize(nr_iters_nnf);
        m_target2source.minimize(nr_iters_nnf);
        if (verbose) std::cerr << "  NNF minimization finished." << std::endl;

        // Instead of upsizing the final target, we build the last target from the next level source image.
        // Thus, the final target is less blurry (see "Space-Time Video Completion" - page 5).
        bool upscaled = false;
        if (level >= 1 && iter_em == nr_iters_em - 1) {
            new_source = m_pyramid[level - 1];
            new_target = target.upsample(new_source.size().width, new_source.size().height, m_pyramid[level - 1].global_mask());
            upscaled = true;
        } else {
            new_source = m_pyramid[level];
            new_target = target.clone();
        }

        auto vote = cv::Mat(new_target.size(), CV_64FC4);
        vote.setTo(cv::Scalar::all(0));

        // Votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence).
        _expectation_step(m_source2target, 1, vote, new_source, upscaled);
        if (verbose) std::cerr << "  Expectation source to target finished." << std::endl;
        _expectation_step(m_target2source, 0, vote, new_source, upscaled);
        if (verbose) std::cerr << "  Expectation target to source finished." << std::endl;

        // Compile votes and update pixel values.
        _maximization_step(new_target, vote);
        if (verbose) std::cerr << "  Minimization step finished." << std::endl;
    }

    return new_target;
}

// Expectation step: vote for best estimations of each pixel.
void Inpainting::_expectation_step(
    const NearestNeighborField &nnf, bool source2target,
    cv::Mat &vote, const MaskedImage &source, bool upscaled
) {
    auto source_size = nnf.source_size();
    auto target_size = nnf.target_size();
    const int patch_size = m_distance_metric->patch_size();

    for (int i = 0; i < source_size.height; ++i) {
        for (int j = 0; j < source_size.width; ++j) {
            if (nnf.source().is_globally_masked(i, j)) continue;
            int yp = nnf.at(i, j, 0), xp = nnf.at(i, j, 1), dp = nnf.at(i, j, 2);
            double w = kDistance2Similarity[dp];

            for (int di = -patch_size; di <= patch_size; ++di) {
                for (int dj = -patch_size; dj <= patch_size; ++dj) {
                    int ys = i + di, xs = j + dj, yt = yp + di, xt = xp + dj;
                    if (!(ys >= 0 && ys < source_size.height && xs >= 0 && xs < source_size.width)) continue;
                    if (nnf.source().is_globally_masked(ys, xs)) continue;
                    if (!(yt >= 0 && yt < target_size.height && xt >= 0 && xt < target_size.width)) continue;
                    if (nnf.target().is_globally_masked(yt, xt)) continue;

                    if (!source2target) {
                        std::swap(ys, yt);
                        std::swap(xs, xt);
                    }

                    if (upscaled) {
                        for (int uy = 0; uy < 2; ++uy) {
                            for (int ux = 0; ux < 2; ++ux) {
                                _weighted_copy(source, 2 * ys + uy, 2 * xs + ux, vote, 2 * yt + uy, 2 * xt + ux, w);
                            }
                        }
                    } else {
                        _weighted_copy(source, ys, xs, vote, yt, xt, w);
                    }
                }
            }
        }
    }
}

// Maximization Step: maximum likelihood of target pixel.
void Inpainting::_maximization_step(MaskedImage &target, const cv::Mat &vote) {
    auto target_size = target.size();
    for (int i = 0; i < target_size.height; ++i) {
        for (int j = 0; j < target_size.width; ++j) {
            const double *source_ptr = vote.ptr<double>(i, j);
            unsigned char *target_ptr = target.get_mutable_image(i, j);

            if (target.is_globally_masked(i, j)) {
                continue;
            }

            if (source_ptr[3] > 0) {
                unsigned char r = cv::saturate_cast<unsigned char>(source_ptr[0] / source_ptr[3]);
                unsigned char g = cv::saturate_cast<unsigned char>(source_ptr[1] / source_ptr[3]);
                unsigned char b = cv::saturate_cast<unsigned char>(source_ptr[2] / source_ptr[3]);
                target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
            } else {
                target.set_mask(i, j, 0);
            }
        }
    }
}

