#ifndef GroundCoordCostFuncs_h
#define GroundCoordCostFuncs_h

#include <vector>

#include <QString>

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "IException.h"

struct GroundError {
  GroundError(std::vector<double> const& observation, std::vector<double> const& weights):
    m_observation(observation), m_weights(weights) {
      // QString msg = "Aborting cost function creation for XYZ point. ";
      // bool potentialNaN = false;
      // if (weights[0] <= 0) {
      //   msg += "X weight is either 0 or negative";
      //   potentialNaN = true;
      // }
      // if (weights[1] <= 0) {
      //   msg += "Y weight is either 0 or negative";
      //   potentialNaN = true;
      // }
      // if (weights[2] <= 0) {
      //   msg += "Z weight is either 0 or negative";
      //   potentialNaN = true;
      // }
      // if (potentialNaN) {
      //   throw IException(IException::Unknown, msg, _FILEINFO_);
      // }
    }

  template <typename T>
  bool operator()(const T* point, T* residuals) const {
    residuals[0] = (point[0] - m_observation[0]) * m_weights[0];
    residuals[1] = (point[1] - m_observation[1]) * m_weights[1];
    residuals[2] = (point[2] - m_observation[2]) * m_weights[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const std::vector<double> & observation,
                                     const std::vector<double> & weights) {
    return (new ceres::AutoDiffCostFunction<GroundError, 3, 3>
            (new GroundError(observation, weights)));
  }

  std::vector<double> m_observation;
  std::vector<double> m_weights;
};

#endif
