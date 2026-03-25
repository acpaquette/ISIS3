#ifndef ReprojectionCostFuncs_h
#define ReprojectionCostFuncs_h

#include <vector>

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/dynamic_cost_function_to_functor.h>

#include "Camera.h"
#include "Displacement.h"
#include "SurfacePoint.h"

struct SnavelyReprojectionFunctor {
  Camera *m_camera;
  int m_positionParamSize;
  int m_rotationParamSize;
  BundleObservationSolveSettings &m_settings;
  
  SnavelyReprojectionFunctor(Camera *camera, 
                             int positionParamSize, 
                             int rotationParamSize,
                             BundleObservationSolveSettings &settings) : 
                            m_camera(camera),  
                            m_positionParamSize(positionParamSize),
                            m_rotationParamSize(rotationParamSize),
                            m_settings(settings) {
  }

  bool operator()(double const* const* parameters, double* results) const {
    if (m_settings.instrumentPointingSolveOption() != 0) {
      std::vector<std::vector<double>> positionPolys(3, std::vector<double>(m_positionParamSize, 0.0));
      SpicePosition *instPosition = m_camera->instrumentPosition();
      // Based on solve settings we need to only update the correct coeffs
      // m_instrumentPositionSolveOption
      // NONE - no updates
      // POSITIONS - 0th element in each polynomial
      // VELOCITEIS - 0th and 1st elements in each polynomial
      // ACCELERATIONS - 0th, 1st and 2nd elements in each polynomial
      // ALL - 0th, 1st and 2nd elements in each polynomial
      for (int j = 0; j < positionPolys.size(); j++) {
        for (int k = 0; k < positionPolys[0].size(); k++) {
          positionPolys[j][k] = parameters[k][j];
        }
      }

      // for (int j = 0; j < positionPolys.size(); j++) {
      //   for (int k = 0; k < positionPolys[0].size(); k++) {
      //     std::cout << positionPolys[j][k] << ", ";
      //   }
      //   std::cout << std::endl;
      // }
      // std::cout << std::endl;
      instPosition->SetPolynomial(positionPolys[0], positionPolys[1], positionPolys[2]);
    }

    if (m_settings.instrumentPointingSolveOption() != 0) {
      std::vector<std::vector<double>> anglePolys(3, std::vector<double>(m_rotationParamSize, 0.0));
      SpiceRotation *instPointing = m_camera->instrumentRotation();
      // Based on solve settings we need to only update the correct coeffs
      // m_instrumentPointingSolveOption
      // NONE - no updates
      // ANGLES - 0th element in each polynomial
      // VELOCITEIS - 0th and 1st elements in each polynomial
      // ACCELERATIONS - 0th, 1st and 2nd elements in each polynomial
      // ALL - 0th, 1st and 2nd elements in each polynomial
      for (int j = 0; j < anglePolys.size(); j++) {
        for (int k = 0; k < anglePolys[0].size(); k++) {
          anglePolys[j][k] = parameters[k + m_positionParamSize][j];
        }
      }
      // for (int j = 0; j < anglePolys.size(); j++) {
      //   for (int k = 0; k < anglePolys[0].size(); k++) {
      //     std::cout << anglePolys[j][k] << ", ";
      //   }
      //   std::cout << std::endl;
      // }
      // std::cout << std::endl;
      instPointing->SetPolynomial(anglePolys[0], anglePolys[1], anglePolys[2]);
    }
    Displacement x(parameters[6][0], Displacement::Units::Kilometers);
    Displacement y(parameters[6][1], Displacement::Units::Kilometers);
    Displacement z(parameters[6][2], Displacement::Units::Kilometers);

    SurfacePoint surfacePoint(x, y, z);
    if (!m_camera->SetGround(surfacePoint)) {
      // Return false if point is not visible
      return false;
    }

    results[0] = m_camera->Sample();
    results[1] = m_camera->Line();

    return true;
  }
};


struct SnavelyReprojectionErrorFunctor {
  SnavelyReprojectionErrorFunctor(double observed_x, double observed_y, Camera *camera, int positionParamSize, int rotationParamSize, BundleObservationSolveSettings &settings)
      : observed_x(observed_x), observed_y(observed_y) {

    auto *cost_function = new ceres::DynamicNumericDiffCostFunction<SnavelyReprojectionFunctor, ceres::CENTRAL>
          (new SnavelyReprojectionFunctor(camera, positionParamSize, rotationParamSize, settings));
    cost_function->AddParameterBlock(positionParamSize);
    cost_function->AddParameterBlock(positionParamSize);
    cost_function->AddParameterBlock(positionParamSize);
    cost_function->AddParameterBlock(rotationParamSize);
    cost_function->AddParameterBlock(rotationParamSize);
    cost_function->AddParameterBlock(rotationParamSize);
    cost_function->AddParameterBlock(3);
    cost_function->SetNumResiduals(2);

    compute_point = std::make_unique<ceres::DynamicCostFunctionToFunctor>(cost_function);
  }

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    T computed[2];
    (*compute_point)(parameters, computed);
    residuals[0] = observed_x - computed[0];
    residuals[1] = observed_y - computed[1];
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static auto* Create(const double observed_x,
                      const double observed_y,
                      Camera *camera,
                      int positionParamSize,
                      int rotationParamSize, 
                      BundleObservationSolveSettings &settings) {
    return new ceres::DynamicNumericDiffCostFunction<SnavelyReprojectionErrorFunctor, ceres::CENTRAL>
           (new SnavelyReprojectionErrorFunctor(observed_x, observed_y, camera, positionParamSize, rotationParamSize, settings));
  }

  double observed_x;
  double observed_y;
  std::unique_ptr<ceres::DynamicCostFunctionToFunctor> compute_point;
};

#endif