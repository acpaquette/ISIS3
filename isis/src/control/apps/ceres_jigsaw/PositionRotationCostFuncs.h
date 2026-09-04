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
    m_positionWeights(positionWeights), m_rotationWeights(rotationWeights) {
        m_originalPolynomials.resize(positionParamSize + rotationParamSize);
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
            residuals[idx] = (cameraPolynomials[i][idx] - m_originalPolynomials[i][idx])/m_positionWeights[i];
        }
    }

    for (size_t i = m_positionWeights.size(); i < m_positionWeights.size() + m_rotationWeights.size(); i++) {
        // Assume 3 (Roll, Pitch, Yaw) based on the ISIS camera polynomials
        for (size_t j = 0; j < 3; j++) {
            size_t idx = (i * 3) + j;
            residuals[idx] = (cameraPolynomials[i][idx] - m_originalPolynomials[i][idx])/m_rotationWeights[i];
        }
    }

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(std::vector<double *> cameraPolynomials,
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
  std::vector<double> m_positionWeights;
  std::vector<double> m_rotationWeights;
};

#endif