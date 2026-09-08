#ifndef PositionRotationCostFuncs_h
#define PositionRotationCostFuncs_h

#include <vector>

#include <QString>

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "IException.h"

struct PositionRotationErrorFunctor {
  PositionRotationErrorFunctor(std::vector<double *> cameraPolynomials,
                               int positionParamSize,
                               int rotationParamSize,
                               const std::vector<double> &positionWeights, 
                               const std::vector<double> &rotationWeights):
    m_positionParamSize(positionParamSize),
    m_rotationParamSize(rotationParamSize),
    m_positionWeights(positionWeights),
    m_rotationWeights(rotationWeights) {
        m_originalPolynomials = std::vector<std::vector<double>>(positionParamSize + rotationParamSize, std::vector<double>(3, 0.0));
        for (int i = 0; i < positionParamSize + rotationParamSize; i++) {
            for (int j = 0; j < 3; j++) {
                m_originalPolynomials[i][j] = cameraPolynomials[i][j];
            }
        }
    }

  template <typename T>
  bool operator()(T const* const* cameraPolynomials, T* residuals) const {

    for (size_t i = 0; i < m_positionWeights.size(); i++) {
        // Assume 3 (X, Y, Z) based on the ISIS camera polynomials
        for (size_t j = 0; j < 3; j++) {
            size_t idx = (i * 3) + j;
            residuals[idx] = (cameraPolynomials[i][j] - m_originalPolynomials[i][j])/m_positionWeights[i];
        }
    }

    for (size_t i = 0; i < m_rotationWeights.size(); i++) {
        // Assume 3 (Roll, Pitch, Yaw) based on the ISIS camera polynomials
        for (size_t j = 0; j < 3; j++) {
            size_t polyIdx = i + m_positionParamSize;
            size_t resIdx = (polyIdx * 3) + j;
            residuals[resIdx] = (cameraPolynomials[polyIdx][j] - m_originalPolynomials[polyIdx][j])/m_rotationWeights[i];
        }
    }

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static auto* Create(std::vector<double *> cameraPolynomials,
                                     int positionParamSize,
                                     int rotationParamSize,
                                     const std::vector<double> &positionWeights, 
                                     const std::vector<double> &rotationWeights) {
    return (new ceres::DynamicAutoDiffCostFunction<PositionRotationErrorFunctor, 4>
            (new PositionRotationErrorFunctor(cameraPolynomials, 
                                              positionParamSize,
                                              rotationParamSize,
                                              positionWeights, 
                                              rotationWeights)));

  }

  std::vector<std::vector<double>> m_originalPolynomials;
  int m_positionParamSize;
  int m_rotationParamSize;
  std::vector<double> m_positionWeights;
  std::vector<double> m_rotationWeights;
};

#endif