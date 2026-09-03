#ifndef PositionRotationCostFuncs_h
#define PositionRotationCostFuncs_h

#include <vector>

#include <QString>

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "IException.h"

struct PositionRotationErrorFunctor {
  PositionRotationErrorFunctor(double const* cameraPolynomials, 
                               int cameraPolySize,
                               const std::vector<double> &positionWeight, 
                               const std::vector<double> &rotationWeight):
    m_positionWeights(positionWeight), m_rotationWeights(rotationWeight) {
        m_originalPolynomials.resize(cameraPolySize);
        for (int i = 0; i < cameraPolySize; i++) {
            m_originalPolynomials[i] = cameraPolynomials[i];
        }
    }

  template <typename T>
  bool operator()(T const* const* cameraPolynomials, T* residuals) const {

    for (size_t i = 0; i < m_positionWeights.size(); i++) {
        // Assume 3 (X, Y, Z) based on the ISIS camera polynomials
        for (size_t j = 0; j < 3; j++) {
            size_t idx = (i * 3) + j;
            residuals[idx] = (cameraPolynomials[0][idx] - m_originalPolynomials[idx])/m_positionWeights[i];
        }
    }

    for (size_t i = 0; i < m_rotationWeights.size(); i++) {
        // Assume 3 (Roll, Pitch, Yaw) based on the ISIS camera polynomials
        for (size_t j = 0; j < 3; j++) {
            size_t idx = (i * 3) + j;
            residuals[idx] = (cameraPolynomials[1][idx] - m_originalPolynomials[idx])/m_rotationWeights[i];
        }
    }

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double *const cameraPolynomials,
                                     int cameraPolySize,
                                     const std::vector<double> &positionWeight, 
                                     const std::vector<double> &rotationWeight) {
    return (new ceres::DynamicAutoDiffCostFunction<PositionRotationErrorFunctor, 4>
            (new PositionRotationErrorFunctor(cameraPolynomials, cameraPolySize, positionWeight, rotationWeight)));

  }

  std::vector<double> m_originalPolynomials;
  std::vector<double> m_positionWeights;
  std::vector<double> m_rotationWeights;
};

#endif