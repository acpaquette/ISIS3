#ifndef GroundCoordCostFuncs_h
#define GroundCoordCostFuncs_h

#include <vector>

#include <QString>

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "IException.h"

struct XYZError {
  XYZError(std::vector<double> const& observation, std::vector<double> const& sigmas):
    m_observation(observation), m_sigmas(sigmas) {
      QString msg = "Aborting cost function creation for XYZ point. ";
      bool potentialNaN = false;
      if (sigmas[0] <= 0) {
        msg += "X sigma is either 0 or negative";
        potentialNaN = true;
      }
      if (sigmas[1] <= 0) {
        msg += "Y sigma is either 0 or negative";
        potentialNaN = true;
      }
      if (sigmas[2] <= 0) {
        msg += "Z sigma is either 0 or negative";
        potentialNaN = true;
      }
      if (potentialNaN) {
        throw IException(IException::Unknown, msg, _FILEINFO_);
      }
    }

  template <typename T>
  bool operator()(const T* point, T* residuals) const {
    residuals[0] = (point[0] - m_observation[0])/m_sigmas[0];
    residuals[1] = (point[1] - m_observation[1])/m_sigmas[1];
    residuals[2] = (point[2] - m_observation[2])/m_sigmas[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const std::vector<double> & observation,
                                     const std::vector<double> & sigmas) {
    return (new ceres::AutoDiffCostFunction<XYZError, 3, 3>
            (new XYZError(observation, sigmas)));
  }

  std::vector<double> m_observation;
  std::vector<double> m_sigmas;
};

#endif
