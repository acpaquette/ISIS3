/** This is free and unencumbered software released into the public domain.

The authors of ISIS do not claim copyright on the contents of this file.
For more details about the LICENSE terms and the AUTHORS, you will
find files of those names at the top level of this repository. **/

/* SPDX-License-Identifier: CC0-1.0 */

#include <iostream>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5Group.hpp>
#include <vector>
#include <cstdio>

#include <QDir>
#include <QList>
#include <QObject>
#include <QSharedPointer>
#include <QString>

#include "Application.h"
#include "Blob.h"
#include "BundleAdjust.h"
#include "BundleObservationSolveSettings.h"
#include "BundleResults.h"
#include "BundleSettings.h"
#include "BundleSolutionInfo.h"
#include "CameraFactory.h"
#include "ControlMeasure.h"
#include "ControlNet.h"
#include "ControlPoint.h"
#include "CubeAttribute.h"
#include "Displacement.h"
#include "GroundCoordCostFuncs.h"
#include "IException.h"
#include "iTime.h"
#include "MaximumLikelihoodWFunctions.h"
#include "PositionRotationCostFuncs.h"
#include "Process.h"
// #include "SensorUtilities.h"
#include "ReprojectionCostFuncs.h"
#include "SerialNumber.h"
#include "SerialNumberList.h"
#include "Table.h"
#include "PvlToJSON.h"
#include "PvlKeyword.h"

#include <ale/Load.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
using json = nlohmann::json;

#include "ceres_jigsaw.h"

using namespace std;
using namespace HighFive;

namespace Isis {
  struct MeasurePartials {
    MeasurePartials(int numTargetPartials = 0) : coeffTarget(2, numTargetPartials),
        coeffPoint3D(2, 3), coeffRHS(2), residualX(0.0), residualY(0.0),
        residualR2ZScore(0.0), observationIndex(-1) { }
    LinearAlgebra::Matrix coeffTarget;
    LinearAlgebra::Matrix coeffImage;
    LinearAlgebra::Matrix coeffPoint3D;
    LinearAlgebra::Vector coeffRHS;
    double residualX;
    double residualY;
    double residualR2ZScore;
    int observationIndex;
  };

  BundleSettingsQsp ceresBundleSettings(UserInterface &ui);
  void ceresCheckImageList(SerialNumberList &heldSerialList, SerialNumberList &cubeSerialList);
  QList<BundleObservationSolveSettings> ceresObservationSolveSettings(UserInterface &ui);
  bool computePartials(MeasurePartials &partials,
                       BundleMeasure &measure,
                       BundleControlPoint &point,
                       QSharedPointer<BundleSettings> settings,
                       BundleResults &bundleResults);

  // 9 potential position coefficients
  // 9 potential rotation coefficients
  // This could change if the user changes the degree of
  // the polynomial to represent rotations/positions
  // const int numParams = 9 + 9;

  struct PointParameters {
    std::vector<double *> parameters;

    PointParameters () {}

    PointParameters(double *polynomials, double *groundPt, int rotationSize, int positionSize) {
      parameters.resize(rotationSize + positionSize + 1);
      for (int i = 0; i < positionSize; i++) {
        parameters[i] = &polynomials[i * 3];
      }

      int positionOffset = 3 * positionSize;
      for (int i = 0; i < rotationSize; i++) {
        parameters[i + positionSize] = &polynomials[positionOffset + (i * 3)];
      }

      parameters[positionSize + rotationSize] = groundPt;
    }
  };

  struct GroundParameters {
    std::vector<double *> parameters;

    GroundParameters () {}

    GroundParameters(double *groundPt) {
      parameters.resize(1);
      parameters[0] = groundPt;
    }
  };

  struct CameraParameters {
    std::vector<double *> parameters;

    CameraParameters () {}

    CameraParameters(double *polynomials, int positionSize, int rotationSize) {
      parameters.resize(positionSize + rotationSize);
      for (int i = 0; i < positionSize; i++) {
        parameters[i] = &polynomials[i * 3];
      }

      int positionOffset = 3 * positionSize;
      for (int i = 0; i < rotationSize; i++) {
        parameters[i + positionSize] = &polynomials[positionOffset + (i * 3)];
      }
    }
  };

  void ceres_jigsaw(UserInterface &ui, Pvl *log) {
    std::cout << std::setprecision(15);
    Progress progress;
    BundleResults bundleResults;
    QSharedPointer<BundleSettings> settings = ceresBundleSettings(ui);

    // initialize solution parameters
    BundleObservationSolveSettings solveSettings = settings->observationSolveSettings(0);
    BundleObservationSolveSettings::InstrumentPositionSolveOption solvePosition = BundleObservationSolveSettings::InstrumentPositionSolveOption::NoPositionFactors;
    BundleObservationSolveSettings::InstrumentPointingSolveOption solveRotation = BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors;
    solvePosition = solveSettings.instrumentPositionSolveOption();
    solveRotation = solveSettings.instrumentPointingSolveOption();
  
    int positionParamSize = solveSettings.spkSolveDegree() + 1;
    int rotationParamSize = solveSettings.ckSolveDegree() + 1;
    std::cout << rotationParamSize << ", " << positionParamSize << std::endl;
    int numParams = rotationParamSize * 3 + positionParamSize * 3;
    std::cout << numParams << std::endl;

    QSharedPointer<ControlNet> network;
    network = QSharedPointer<ControlNet>(new ControlNet(ui.GetFileName("CNET"), &progress, settings->controlPointCoordTypeReports()));
    SerialNumberList snList(ui.GetFileName("FROMLIST"));
    network->SetImages(snList, &progress);
    std::map<QString, double *> polyMap;
    std::map<QString, double *> originalPolyMap;
    std::vector<CameraParameters> bundleCameraParameters(snList.size());
    BundleObservationVector bundleObservations;
    // This is set globally for all cameras, there should likely be a way to set this on an individual
    // camera basis
    std::vector<double> positionWeights(solveSettings.aprioriPositionSigmas().constBegin(), 
                                        solveSettings.aprioriPositionSigmas().constEnd());
    if (positionWeights.size() < positionParamSize) {
      for (int i = positionWeights.size(); i < positionParamSize; i++) {
        positionWeights.push_back(1);
      }
    }
    for (int i = 0; i < positionWeights.size(); i++) {
      if (positionWeights[i] != 0 ) {
        double posWeight = positionWeights[i];
        // Convert to km
        posWeight = 1.0 / (posWeight * posWeight * 1.0e-3 * 1.0e-3);
        positionWeights[i] = posWeight;
      }
    }

    std::vector<double> rotationWeights(solveSettings.aprioriPointingSigmas().constBegin(), 
                                        solveSettings.aprioriPointingSigmas().constEnd());
    if (rotationWeights.size() < rotationParamSize) {
      for (int i = rotationWeights.size(); i < rotationParamSize; i++) {
        rotationWeights.push_back(1);
      }
    }
    for (int i = 0; i < rotationWeights.size(); i++) {
      if (rotationWeights[i] != 0 ) {
        double rotWeight = rotationWeights[i];
        // Convert to radians
        rotWeight = 1.0 / (rotWeight * rotWeight * DEG2RAD * DEG2RAD);
        rotationWeights[i] = rotWeight;
      }
    }

    for (int i = 0; i < snList.size(); i++) {
      QString observationNumber = snList.observationNumber(i);
      QString instrumentId = snList.spacecraftInstrumentId(i);
      QString serialNumber = snList.serialNumber(i);
      QString fileName = snList.fileName(i);
      Camera *camera = network->Camera(serialNumber);

      // create a new BundleImage and add to new (or existing if observation mode is on)
      // BundleObservation
      BundleImageQsp image = BundleImageQsp(new BundleImage(camera, serialNumber, fileName));

      if (!image) {
        QString msg = "In BundleAdjust::init(): image " + fileName + "is null." + "\n";
        throw IException(IException::Programmer, msg, _FILEINFO_);
      }

      BundleObservationQsp observation =
          bundleObservations.addNew(image, observationNumber, instrumentId, settings);

      if (!observation) {
        QString msg = "In BundleAdjust::init(): observation "
                      + observationNumber + "is null." + "\n";
        throw IException(IException::Programmer, msg, _FILEINFO_);
      }

      double *cameraPolynomials = new double[numParams];
      double *originalCameraPolynomials = new double[numParams];
      for (int j = 0; j < numParams; j++) {
        cameraPolynomials[j] = 0.0;
        originalCameraPolynomials[j] = 0.0;
      }
      if (solvePosition != BundleObservationSolveSettings::InstrumentPositionSolveOption::NoPositionFactors) {
        SpicePosition *spicePosition = camera->instrumentPosition();
        
        // first, set the degree of the spk polynomial to be fit for a priori values
        spicePosition->SetPolynomialDegree(solveSettings.spkDegree());

        // now, set what kind of interpolation to use (polynomial function or
        // polynomial function over hermite spline)
        spicePosition->SetPolynomial(solveSettings.positionInterpolationType());

        // finally, set the degree of the position polynomial actually used in the bundle adjustment
        spicePosition->SetPolynomialDegree(solveSettings.spkSolveDegree());
        std::vector<std::vector<double>> positionPolys(3, std::vector<double>(positionParamSize, 0.0));
        spicePosition->GetPolynomial(positionPolys[0], positionPolys[1], positionPolys[2]);
        for (int j = 0; j < positionPolys.size(); j++) {
          for (int k = 0; k < positionPolys[j].size(); k++) {
            cameraPolynomials[(k * positionPolys.size() + j)] = positionPolys[j][k];
            originalCameraPolynomials[(k * positionPolys.size() + j)] = positionPolys[j][k];
          }
        }
      }

      if (solveRotation != BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors) {
        SpiceRotation *spiceRotation = camera->instrumentRotation();

        // first, set the degree of the polynomial to be fit for a priori values
        spiceRotation->SetPolynomialDegree(solveSettings.ckDegree());

        // now, set what kind of interpolation to use (polynomial function or
        // polynomial function over a pointing cache)
        spiceRotation->SetPolynomial(solveSettings.pointingInterpolationType());

        // finally, set the degree of the pointing polynomial actually used in the bundle adjustment
        spiceRotation->SetPolynomialDegree(solveSettings.ckSolveDegree());
        std::vector<std::vector<double>> anglePolys(3, std::vector<double>(rotationParamSize, 0.0));
        spiceRotation->GetPolynomial(anglePolys[0], anglePolys[1], anglePolys[2]);
        int positionOffset = 3 * positionParamSize;
        for (int j = 0; j < anglePolys.size(); j++) {
          for (int k = 0; k < anglePolys[j].size(); k++) {
            cameraPolynomials[positionOffset + (k * anglePolys.size() + j)] = anglePolys[j][k];
            originalCameraPolynomials[positionOffset + (k * anglePolys.size() + j)] = anglePolys[j][k];
          }
        }
      }

      polyMap[serialNumber] = cameraPolynomials;
      originalPolyMap[serialNumber] = originalCameraPolynomials;
      bundleCameraParameters[i] = CameraParameters(polyMap[serialNumber], 
                                                   positionParamSize,
                                                   rotationParamSize);
    }

    // Convert to pointer of similar data
    // That is, each entry should point to some cameras set of
    // polys rather than making a duplicate entry
    std::vector<PointParameters> bundlePointParameters(network->GetNumValidMeasures());
    std::vector<GroundParameters> bundleGroundParameters(network->GetNumValidPoints());
    double **observedPoints = new double*[network->GetNumValidMeasures()];

    std::vector<Camera *> cameras(network->GetNumValidMeasures(), nullptr);
    std::vector<double> measureWeights(network->GetNumValidMeasures(), 1.0);
    std::vector<std::vector<double>> observedGround(network->GetNumValidPoints(), std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> groundWeights(network->GetNumValidPoints(), std::vector<double>(3, 1.0));
    int point_idx = 0;
    int measure_idx = 0;
    progress.SetText("Loading ceres data...");
    int numNetworkPoints = network->GetNumPoints();
    progress.SetMaximumSteps(numNetworkPoints);
    progress.CheckStatus();
    int numConstrainedCoordinates = 0;
    int numObservations = 0;

    for (int i = 0; i < numNetworkPoints; i++) {
      if (network->GetPoint(i)->IsIgnored()) {
        continue;
      }
      BundleControlPoint point = BundleControlPoint(settings, network->GetPoint(i));
      point.rawControlPoint()->ComputeApriori();
      
      SurfacePoint ground = point.rawControlPoint()->GetAprioriSurfacePoint();
      double *groundCoord = new double[3];
      SurfacePoint::CoordinateType coordType = settings->controlPointCoordTypeBundle();
      if (coordType == SurfacePoint::Latitudinal) {
        groundCoord[0] = ground.GetLatitude().radians();
        groundCoord[1] = ground.GetLongitude().radians();
        groundCoord[2] = ground.GetLocalRadius().kilometers();
      }
      else {
        groundCoord[0] = ground.GetX().kilometers();
        groundCoord[1] = ground.GetY().kilometers();
        groundCoord[2] = ground.GetZ().kilometers();
      }
      
      bundleGroundParameters[point_idx] = GroundParameters(groundCoord);
      observedGround[point_idx] = {groundCoord[0], groundCoord[1], groundCoord[2]};
      groundWeights[point_idx] = std::vector<double>(point.weights().begin(), point.weights().end());
      for (int j = 0; j < groundWeights[point_idx].size(); j++) {
        if (!IsSpecial(groundWeights[point_idx][j]) && groundWeights[point_idx][j] > 0.0) {
          numConstrainedCoordinates++;
        }
      }

      for (int j = 0; j < point.numberOfMeasures(); j++) {
        numObservations += 2;
        QSharedPointer<BundleMeasure> measure = point.at(j);

        observedPoints[measure_idx] = new double[2];
        observedPoints[measure_idx][0] = measure->sample();
        observedPoints[measure_idx][1] = measure->line();

        QString serialNumber = measure->cubeSerialNumber();
        cameras[measure_idx] = measure->camera();
        
        // Magic number from ISIS implementation. This does not actually set the sigma to 1.4
        // See function for more detail
        measure->setSigma(1.4);
        // Do we want weight or sigma?
        // try with sigma, see what happens
        measureWeights[measure_idx] = measure->weight();

        // Copy data to spots in pointParameters
        bundlePointParameters[measure_idx] = PointParameters(polyMap[serialNumber], groundCoord, rotationParamSize, positionParamSize);
        measure_idx++;
      }
      point_idx++;
      progress.CheckStatus();
    }

    bundleResults.setNumberConstrainedPointParameters(numConstrainedCoordinates);
    bundleResults.setNumberImageObservations(numObservations);
    bundleResults.setNumberUnknownParameters(numParams + 3 * numNetworkPoints);

    std::cout << "INITIAL Polys: \n";
    for ( int i = 0; i < 10; i++) {
      int j = 0;
      for (;j < positionParamSize; j++) {
        for (int k = 0; k < 3; k++) {
          std::cout << bundlePointParameters[i].parameters[j][k] << ", ";
        }
      }
      std::cout << std::endl;

      j = positionParamSize;
      for (;j < (positionParamSize + rotationParamSize); j++) {
        for (int k = 0; k < 3; k++) {
          std::cout << bundlePointParameters[i].parameters[j][k] << ", ";
        }
      }
      std::cout << std::endl;
    }

    std::cout << "INITIAL GP: \n";
    for ( int i = 0; i < 10; i++) {
      std::cout << bundleGroundParameters[i].parameters[0][0] << ", ";
      std::cout << bundleGroundParameters[i].parameters[0][1] << ", ";
      std::cout << bundleGroundParameters[i].parameters[0][2] << std::endl;
    }

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    progress.SetText("Loading ceres camera polygon cost functions...");
    int snListSize = snList.size();
    progress.SetMaximumSteps(snListSize);
    progress.CheckStatus();
    std::vector<ceres::ResidualBlockId> cameraResidualIds(snListSize, nullptr);
    for (int i = 0; i < snListSize; i++) {
      auto* cost_function = PositionRotationErrorFunctor::Create(bundleCameraParameters[i].parameters,
                                                                 positionParamSize,
                                                                 rotationParamSize,
                                                                 positionWeights,
                                                                 rotationWeights);
      for (int j = 0; j < positionParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      for (int j = 0; j < positionParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      cost_function->SetNumResiduals((positionWeights.size() + rotationWeights.size()) * 3);
      cameraResidualIds[i] = problem.AddResidualBlock(cost_function, loss_function, bundleCameraParameters[i].parameters);

      if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocityAcceleration) {
        problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[2]);
        if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocity) {
          problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[1]);
          if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionOnly) {
            problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[0]);
          }
        }
      }

      if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocityAcceleration) {
        problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[5]);
        if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocity) {
          problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[4]);
          if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesOnly) {
            problem.SetParameterBlockConstant(bundleCameraParameters[i].parameters[3]);
          }
        }
      }
      progress.CheckStatus();
    }

    progress.SetText("Loading ceres projection cost functions...");
    int numValidMeasures = network->GetNumValidMeasures();
    progress.SetMaximumSteps(numValidMeasures);
    progress.CheckStatus();
    std::vector<ceres::ResidualBlockId> projectionResidualIds(numValidMeasures, nullptr);
    SurfacePoint::CoordinateType coordType = settings->controlPointCoordTypeBundle();
    for (int i = 0; i < numValidMeasures; ++i) {
      auto* cost_function =
          SnavelyReprojectionErrorFunctor::Create(observedPoints[i][0],
                                                  observedPoints[i][1],
                                                  cameras[i],
                                                  positionParamSize,
                                                  rotationParamSize,
                                                  measureWeights[i],
                                                  coordType,
                                                  &solveSettings);
      for (int j = 0; j < positionParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      for (int j = 0; j < rotationParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      cost_function->AddParameterBlock(3);
      cost_function->SetNumResiduals(2);
      projectionResidualIds[i] = problem.AddResidualBlock(cost_function,
                                                          loss_function,
                                                          bundlePointParameters[i].parameters);
      // I'm not sure if we need this if the parameters are constrained in the camera loop
      // They point to the same set of polynomials
      // if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocityAcceleration) {
      //   problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[2]);
      //   if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocity) {
      //     problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[1]);
      //     if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionOnly) {
      //       problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[0]);
      //     }
      //   }
      // }

      // if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocityAcceleration) {
      //   problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[5]);
      //   if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocity) {
      //     problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[4]);
      //     if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesOnly) {
      //       problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[3]);
      //     }
      //   }
      // }
      progress.CheckStatus();
    }

    progress.SetText("Loading ceres ground functions...");
    int numValidPoints = network->GetNumValidPoints();
    progress.SetMaximumSteps(numValidPoints);
    progress.CheckStatus();
    std::vector<ceres::ResidualBlockId> groundResidualIds(numValidPoints, nullptr);
    for (int i = 0; i < numValidPoints; i++) {
      auto* cost_function = GroundError::Create(observedGround[i], groundWeights[i]);
      groundResidualIds[i] = problem.AddResidualBlock(cost_function, loss_function, bundleGroundParameters[i].parameters);
      progress.CheckStatus();
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = ui.GetDouble("SIGMA0");
    options.max_num_iterations = ui.GetInteger("MAXITS");
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // Update the networks measures and points then write to the new output network
    point_idx = 0;
    measure_idx = 0;

    for (int i = 0; i < numNetworkPoints; i++) {
      if (network->GetPoint(i)->IsIgnored()) {
        continue;
      }
      ControlPoint *point = network->GetPoint(i);
      double *groundPoint = bundleGroundParameters[point_idx].parameters[0];
      SurfacePoint adjustedSurfacePoint;
      SurfacePoint::CoordinateType coordType = settings->controlPointCoordTypeBundle();
      if (coordType == SurfacePoint::Latitudinal) {
        adjustedSurfacePoint.SetSpherical(Latitude(groundPoint[0], Angle::Units::Radians), 
                                          Longitude(groundPoint[1], Angle::Units::Radians),
                                          Distance(groundPoint[2], Distance::Units::Kilometers));
      }
      else {
        adjustedSurfacePoint.SetRectangular(Distance(groundPoint[0], Distance::Units::Kilometers), 
                                            Distance(groundPoint[1], Distance::Units::Kilometers),
                                            Distance(groundPoint[2], Distance::Units::Kilometers));
      }
      point->SetAdjustedSurfacePoint(adjustedSurfacePoint);
      for (int j = 0; j < point->GetNumMeasures(); j++) {
        ControlMeasure *measure = point->GetMeasure(j);
        if (measure->IsIgnored()) {
          continue;
        }
        ceres::ResidualBlockId resId = projectionResidualIds[measure_idx];
        double residuals[2];
        problem.EvaluateResidualBlock(resId, true, nullptr, residuals, nullptr);
        measure->SetResidual(residuals[0], residuals[1]);

        measure_idx++;
      }
      point_idx++;
    }

    for (int i = 0; i < snList.size(); i++) {
      QString serialNumber = snList.serialNumber(i);
      QSharedPointer<BundleObservation> observation = bundleObservations.observationByCubeSerialNumber(serialNumber);
      LinearAlgebra::Vector &corrections = observation->parameterCorrections();
      int nCameraPositionCoefficients = solveSettings.numberCameraPositionCoefficientsSolved();
      int nCameraAngleCoefficients = solveSettings.numberCameraAngleCoefficientsSolved();

      int nParameters = 3*nCameraPositionCoefficients + 2*nCameraAngleCoefficients;
      if (nCameraAngleCoefficients >= 1 && solveSettings.solveTwist()) {
        nParameters += nCameraAngleCoefficients;
      }

      corrections.resize(nParameters);
      int j = 0;
      int polyMapIdx = 0;
      for (; j < nCameraPositionCoefficients; j++) {
        for (int k = 0; k < 3; k++) {
          corrections[k + (j * 3)] = polyMap[serialNumber][(positionParamSize * k) + polyMapIdx];
        }
        polyMapIdx++;
      }

      polyMapIdx = positionParamSize * 3;
      j = nCameraPositionCoefficients;
      for (; j < nCameraPositionCoefficients + nCameraAngleCoefficients; j++) {
        for (int k = 0; k < 3; k++) {
          corrections[k + (j * 3)] = polyMap[serialNumber][(rotationParamSize * k) + polyMapIdx];
        }
        polyMapIdx++;
      }
    }

    int numberImages = snList.size();
    QVector<Statistics> rmsImageSampleResiduals(numberImages);
    QVector<Statistics> rmsImageLineResiduals(numberImages);
    QVector<Statistics> rmsImageResiduals(numberImages);

    // number of target body partials per measure (note this does not change through the adjustment)
    int numTargetPartials = 0;
    if (settings->solveTargetBody()) {
      // TODO make sure numTargetBodyParameters is greater than 0
      numTargetPartials = settings->numberTargetBodyParameters();
    }

    // reused by every measure, so its matrices are allocated once for the whole adjustment
    MeasurePartials partials(numTargetPartials);
    QVector<QSharedPointer<BundleControlPoint>> bundleControlPoints;
  
    point_idx = 0;
    measure_idx = 0;
    for (int i = 0; i < numNetworkPoints; i++) {

      if (network->GetPoint(i)->IsIgnored()) {
        continue;
      }

      QSharedPointer<BundleControlPoint> point(new BundleControlPoint(settings, network->GetPoint(i)));
      bundleControlPoints.append(point);

      double *groundPoint = bundleGroundParameters[point_idx].parameters[0];
      SurfacePoint adjustedSurfacePoint;
      SurfacePoint::CoordinateType coordType = settings->controlPointCoordTypeBundle();
      if (coordType == SurfacePoint::Latitudinal) {
        adjustedSurfacePoint.SetSpherical(Latitude(groundPoint[0], Angle::Units::Radians), 
                                          Longitude(groundPoint[1], Angle::Units::Radians),
                                          Distance(groundPoint[2], Distance::Units::Kilometers));
      }
      else {
        adjustedSurfacePoint.SetRectangular(Distance(groundPoint[0], Distance::Units::Kilometers), 
                                            Distance(groundPoint[1], Distance::Units::Kilometers),
                                            Distance(groundPoint[2], Distance::Units::Kilometers));
      }
      point->setAdjustedSurfacePoint(adjustedSurfacePoint);

      boost::numeric::ublas::bounded_vector< double, 3 > &corrections = point->corrections();
      if (coordType == SurfacePoint::Latitudinal) {
        corrections[0] = point->rawControlPoint()->GetAprioriSurfacePoint().GetLatitude().radians() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetLatitude().radians();
        corrections[1] = point->rawControlPoint()->GetAprioriSurfacePoint().GetLongitude().radians() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetLongitude().radians();
        corrections[2] = point->rawControlPoint()->GetAprioriSurfacePoint().GetLocalRadius().kilometers() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetLocalRadius().kilometers();
      }
      else {
        corrections[0] = point->rawControlPoint()->GetAprioriSurfacePoint().GetX().kilometers() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetX().kilometers();
        corrections[1] = point->rawControlPoint()->GetAprioriSurfacePoint().GetY().kilometers() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetY().kilometers();
        corrections[2] = point->rawControlPoint()->GetAprioriSurfacePoint().GetZ().kilometers() - point->rawControlPoint()->GetAdjustedSurfacePoint().GetZ().kilometers();
      }


      if (point->isRejected()) {
        continue;
      }

      for (int j = 0; j < point->numberOfMeasures(); j++) {
        QSharedPointer<BundleMeasure> measure = point->at(j);
        QString serialNumber = measure->cubeSerialNumber();
        QSharedPointer<BundleObservation> observation = bundleObservations.observationByCubeSerialNumber(serialNumber);
        QSharedPointer<BundleImage> image = observation->imageByCubeSerialNumber(serialNumber);
        measure->setParentObservation(observation);
        measure->setParentImage(image);
        computePartials(partials, *measure, *point, settings, bundleResults);

        bundleResults.addResidualsProbabilityDistributionObservation(partials.residualX);
        bundleResults.addResidualsProbabilityDistributionObservation(partials.residualY);

        if (bundleResults.numberMaximumLikelihoodModels()
              > bundleResults.maximumLikelihoodModelIndex()) {
          // Dynamically build the cumulative probability distribution of the R^2 residual Z Scores
          bundleResults.addProbabilityDistributionObservation(partials.residualR2ZScore);
        }
        ceres::ResidualBlockId resId = projectionResidualIds[measure_idx++];
        double residuals[2];
        problem.EvaluateResidualBlock(resId, true, nullptr, residuals, nullptr);
        // measure->SetResidual(residuals[0], residuals[1]);
        double sampleResidual = fabs(residuals[0]);
        double lineResidual = fabs(residuals[1]);
        int imageIndex = snList.serialNumberIndex(measure->cubeSerialNumber());
        rmsImageSampleResiduals[imageIndex].AddData(sampleResidual);
        rmsImageLineResiduals[imageIndex].AddData(lineResidual);
        rmsImageResiduals[imageIndex].AddData(lineResidual);
        rmsImageResiduals[imageIndex].AddData(sampleResidual);
      }
      point_idx++;
    }

    bundleResults.setObservations(bundleObservations);
    bundleResults.setBundleControlPoints(bundleControlPoints);
    bundleResults.setRmsImageResidualLists(rmsImageLineResiduals.toList(),
                                           rmsImageSampleResiduals.toList(),
                                           rmsImageResiduals.toList());
    bundleResults.setOutputControlNet(network);
    BundleSolutionInfo bundleSolutionInfo(settings,
                                          ui.GetFileName("CNET"),
                                          FileName(""),
                                          bundleResults,
                                          QList<ImageList *>{});
    bundleSolutionInfo.setOutputControlName( FileName(ui.GetFileName("ONET")).expanded() );
    cout << "\nGenerating report files\n" << endl;

    // write output files
    if (ui.GetBoolean("BUNDLEOUT_TXT")) {
      bundleSolutionInfo.outputText();
    }

    // if (ui.GetBoolean("IMAGESCSV")) {
    //   bundleSolutionInfo->outputImagesCSV();
    // }

    // if (ui.GetBoolean("OUTPUT_CSV")) {
    //   bundleSolutionInfo->outputPointsCSV();
    // }
    // if (ui.GetBoolean("RESIDUALS_CSV")) {
    //   bundleSolutionInfo->outputResiduals();
    // }

    // // write lidar csv output file
    // if (ui.GetBoolean("LIDAR_CSV")) {
    //   bundleSolutionInfo->outputLidarCSV();
    // }

    // write updated control net
    // bundleAdjustment->controlNet()->Write(ui.GetFileName("ONET"));

    // write updated lidar data file
    // if (ui.WasEntered("LIDARDATA")) {
    //   if (ui.GetString("OLIDARFORMAT") == "JSON") {
    //     bundleAdjustment->lidarData()->write(ui.GetFileName("OLIDARDATA"),LidarData::Format::Json);
    //   }
    //   else {
    //     bundleAdjustment->lidarData()->write(ui.GetFileName("OLIDARDATA"),LidarData::Format::Binary);
    //   }
    // }

    std::cout << "POST Polys: \n";
    for ( int i = 0; i < 10; i++) {
      int j = 0;
      for (;j < positionParamSize; j++) {
        for (int k = 0; k < 3; k++) {
          std::cout << bundleCameraParameters[i].parameters[j][k] << ", ";
        }
      }
      std::cout << std::endl;

      j = positionParamSize;
      for (;j < (positionParamSize + rotationParamSize); j++) {
        for (int k = 0; k < 3; k++) {
          std::cout << bundleCameraParameters[i].parameters[j][k] << ", ";
        }
      }
      std::cout << std::endl;
    }

    std::cout << "POST GP Diff: \n";
    for ( int i = 0; i < 20; i++) {
      std::cout << bundleGroundParameters[i].parameters[0][0] - observedGround[i][0] << ", ";
      std::cout << bundleGroundParameters[i].parameters[0][1] - observedGround[i][1] << ", ";
      std::cout << bundleGroundParameters[i].parameters[0][2] - observedGround[i][2] << std::endl;
    }

    for (int i = 0; i < snList.size(); i++) {
      QString serialNumber = snList.serialNumber(i);
      QString fileName = snList.fileName(i);
      Cube cube(snList.fileName(i));
      Camera *originalCamera = cube.camera();
      Camera *networkCamera = network->Camera(serialNumber);
      // compare Angles of original camera against new angles
      const std::vector<double> networkCoord = networkCamera->instrumentPosition()->GetCenterCoordinate();
      const std::vector<double> networkAngles = networkCamera->instrumentRotation()->GetCenterAngles();

      const std::vector<double> originalCoord = originalCamera->instrumentPosition()->GetCenterCoordinate();
      const std::vector<double> originalAngles = originalCamera->instrumentRotation()->GetCenterAngles();
      std::cout << serialNumber << std::endl;
      std::cout << snList.fileName(serialNumber) << std::endl;

      std::cout << originalCoord[0] << ", ";
      std::cout << originalCoord[1] << ", ";
      std::cout << originalCoord[2] << std::endl << std::endl;
      std::cout << networkCoord[0] - originalCoord[0] << ", ";
      std::cout << networkCoord[1] - originalCoord[1] << ", ";
      std::cout << networkCoord[2] - originalCoord[2] << std::endl << std::endl;
      std::cout << networkCoord[0] << ", ";
      std::cout << networkCoord[1] << ", ";
      std::cout << networkCoord[2] << std::endl << std::endl;

      std::cout << originalAngles[0] << ", ";
      std::cout << originalAngles[1] << ", ";
      std::cout << originalAngles[2] << std::endl << std::endl;
      std::cout << networkAngles[0] - originalAngles[0] << ", ";
      std::cout << networkAngles[1] - originalAngles[1] << ", ";
      std::cout << networkAngles[2] - originalAngles[2] << std::endl << std::endl;
      std::cout << networkAngles[0] << ", ";
      std::cout << networkAngles[1] << ", ";
      std::cout << networkAngles[2] << std::endl << std::endl;

      for (int j = 0; j < numParams; j++) {
        std::cout << polyMap[serialNumber][j] - originalPolyMap[serialNumber][j] << std::endl;
      }
    }

    if (ui.GetBoolean("UPDATE") ) {
      QString jigComment = "Jigged = " + Isis::iTime::CurrentLocalTime();
      // Loop through images
      for (int i = 0; i < snList.size(); i++) {
        Process p;
        CubeAttributeInput inAtt;
        Cube *cube = p.SetInputCube(snList.fileName(i), inAtt, ReadWrite);
        //check for existing polygon, if exists delete it
        if (cube->label()->hasObject("Polygon")) {
          cube->label()->deleteObject("Polygon");
        }

        // check for CameraStatistics Table, if exists, delete
        for (int iobj = 0; iobj < cube->label()->objects(); iobj++) {
          PvlObject obj = cube->label()->object(iobj);
          if (obj.name() != "Table") continue;
          if (obj["Name"][0] != QString("CameraStatistics")) continue;
          cube->label()->deleteObject(iobj);
          break;
        }
        double *updatedCameraPolynomials = polyMap[snList.serialNumber(i)];

        // Write bundle adjustment values to cube
        Camera *camera = cube->camera();
        if (solvePosition != BundleObservationSolveSettings::InstrumentPositionSolveOption::NoPositionFactors) {
          std::vector<std::vector<double>> positionPolys(positionParamSize, std::vector<double>(positionParamSize, 0.0));
          for (int j = 0; j < positionPolys[0].size(); j++) {
            for (int k = 0; k < positionPolys.size(); k++) {
              positionPolys[k][j] = updatedCameraPolynomials[(j * positionParamSize) + k];
            }
          }
          camera->instrumentPosition()->SetPolynomialDegree(solveSettings.spkSolveDegree());
          camera->instrumentPosition()->SetPolynomial(positionPolys[0], 
                                                      positionPolys[1], 
                                                      positionPolys[2],
                                                      solveSettings.positionInterpolationType());
          Table spvector = camera->instrumentPosition()->Cache("InstrumentPosition");
          spvector.Label().addComment(jigComment);
          cube->write(spvector);
        }

        if (solveRotation != BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors) {
          std::vector<std::vector<double>> anglePolys(positionParamSize, std::vector<double>(positionParamSize, 0.0));
          int positionOffset = 3 * positionParamSize;
          for (int j = 0; j < anglePolys[0].size(); j++) {
            for (int k = 0; k < anglePolys.size(); k++) {
              anglePolys[k][j] = updatedCameraPolynomials[(positionOffset + j * rotationParamSize) + k];
            }
          }
          camera->instrumentRotation()->SetPolynomialDegree(solveSettings.ckSolveDegree());
          camera->instrumentRotation()->SetPolynomial(anglePolys[0], 
                                                      anglePolys[1], 
                                                      anglePolys[2], 
                                                      solveSettings.pointingInterpolationType());
          Table cmatrix = camera->instrumentRotation()->Cache("InstrumentPointing");
          cmatrix.Label().addComment(jigComment);
          cube->write(cmatrix);
        }
      }
    }

    for (auto iterator = polyMap.begin(); iterator != polyMap.end(); iterator++ ) {
      delete [](iterator->second);
    }
    polyMap.clear();

    for (auto iterator = originalPolyMap.begin(); iterator != originalPolyMap.end(); iterator++ ) {
      delete [](iterator->second);
    }
    originalPolyMap.clear();

    int groundPointIdx = 0;
    for (int i = 0; i < network->GetNumPoints(); i++) {
      ControlPoint *point = network->GetPoint(i);
      if (point->IsIgnored()) {
        continue;
      }
      delete []bundlePointParameters[groundPointIdx].parameters[rotationParamSize + positionParamSize];
      for (int j = 0; j < point->GetNumMeasures(); j++) {
        ControlMeasure *measure = point->GetMeasure(j);
        if (measure->IsIgnored()) {
          continue;
        }
        delete []observedPoints[groundPointIdx];
        groundPointIdx++;
      }
    }

    delete []observedPoints;
  }

  BundleSettingsQsp ceresBundleSettings(UserInterface &ui) {
    //  BundleSettings settings;
    BundleSettingsQsp settings = BundleSettingsQsp(new BundleSettings);

    settings->setValidateNetwork(true);

    // solve options
    QString coordTypeBundleStr = ui.GetString("CONTROL_POINT_COORDINATE_TYPE_BUNDLE");
    QString coordTypeReportsStr = ui.GetString("CONTROL_POINT_COORDINATE_TYPE_REPORTS");
    SurfacePoint::CoordinateType ctypeBundle = SurfacePoint::Latitudinal;
    SurfacePoint::CoordinateType ctypeReports = SurfacePoint::Latitudinal;

    if (coordTypeBundleStr == "RECTANGULAR") {
      ctypeBundle = SurfacePoint::Rectangular;
    }

    if (coordTypeReportsStr == "RECTANGULAR") {
      ctypeReports = SurfacePoint::Rectangular;
    }

    double coord1Sigma = Isis::Null;
    double coord2Sigma = Isis::Null;
    double coord3Sigma = Isis::Null;
    if (ui.WasEntered("POINT_LATITUDE_SIGMA")) {
      coord1Sigma = ui.GetDouble("POINT_LATITUDE_SIGMA");
    }
    if (ui.WasEntered("POINT_LONGITUDE_SIGMA")) {
      coord2Sigma = ui.GetDouble("POINT_LONGITUDE_SIGMA");
    }
    if (ui.WasEntered("POINT_RADIUS_SIGMA")) {
      coord3Sigma = ui.GetDouble("POINT_RADIUS_SIGMA");
    }
    if (ui.WasEntered("POINT_X_SIGMA")) {
      coord1Sigma = ui.GetDouble("POINT_X_SIGMA");
    }
    if (ui.WasEntered("POINT_Y_SIGMA")) {
      coord2Sigma = ui.GetDouble("POINT_Y_SIGMA");
    }
    if (ui.WasEntered("POINT_Z_SIGMA")) {
      coord3Sigma = ui.GetDouble("POINT_Z_SIGMA");
    }

    settings->setSolveOptions(ui.GetBoolean("OBSERVATIONS"),
                              ui.GetBoolean("UPDATE"),
                              ui.GetBoolean("ERRORPROPAGATION"),
                              ui.GetBoolean("RADIUS"),
                              ctypeBundle, ctypeReports,
                              coord1Sigma,
                              coord2Sigma,
                              coord3Sigma);

    // Don't create the inverse correlation matrix file
    settings->setCreateInverseMatrix(false);

    settings->setOutlierRejection(ui.GetBoolean("OUTLIER_REJECTION"),
                                 ui.GetDouble("REJECTION_MULTIPLIER"));

    QList<BundleObservationSolveSettings> solveSettingsList = ceresObservationSolveSettings(ui);
    settings->setObservationSolveOptions(solveSettingsList);
    // convergence criteria
    settings->setConvergenceCriteria(BundleSettings::Sigma0,
                                    ui.GetDouble("SIGMA0"),
                                    ui.GetInteger("MAXITS"));

    // max likelihood estimation
    if (ui.GetString("MODEL1").compare("NONE") != 0) {
      // if model1 is not "NONE", add to the models list with its quantile
      settings->addMaximumLikelihoodEstimatorModel(
          MaximumLikelihoodWFunctions::stringToModel(ui.GetString("MODEL1")),
              ui.GetDouble("MAX_MODEL1_C_QUANTILE"));

      if (ui.GetString("MODEL2").compare("NONE") != 0) {
        // if model2 is not "NONE", add to the models list with its quantile
        settings->addMaximumLikelihoodEstimatorModel(
            MaximumLikelihoodWFunctions::stringToModel(ui.GetString("MODEL2")),
                ui.GetDouble("MAX_MODEL2_C_QUANTILE"));

        if (ui.GetString("MODEL3").compare("NONE") != 0) {
          // if model3 is not "NONE", add to the models list with its quantile
          settings->addMaximumLikelihoodEstimatorModel(
              MaximumLikelihoodWFunctions::stringToModel(ui.GetString("MODEL3")),
                  ui.GetDouble("MAX_MODEL3_C_QUANTILE"));
        }
      }
    }

    // target body options
    if (ui.GetBoolean("SOLVETARGETBODY") == true) {
      PvlObject obj;
      ui.GetFileName("TBPARAMETERS");
      Pvl tbParPvl(FileName(ui.GetFileName("TBPARAMETERS")).expanded());
      if (!tbParPvl.hasObject("Target")) {
        QString msg = "Input Target parameters file missing main Target object";
        throw IException(IException::User, msg, _FILEINFO_);
      }

      // read target body pvl file into BundleTargetBody object
      BundleTargetBodyQsp bundleTargetBody = BundleTargetBodyQsp(new BundleTargetBody);

      obj = tbParPvl.findObject("Target");
      bundleTargetBody->readFromPvl(obj);

      // ensure user entered something to adjust
      if (bundleTargetBody->numberParameters() == 0) {
        string msg = "Must solve for at least one target body option";
        throw IException(IException::User, msg, _FILEINFO_);
      }

      settings->setBundleTargetBody(bundleTargetBody);
    }

    // output options
    QString outputfileprefix = "";
    if (ui.WasEntered("FILE_PREFIX"))  {
      outputfileprefix = ui.GetString("FILE_PREFIX");
      int length = (outputfileprefix.length()) - 1;
      QChar endvalue = outputfileprefix[length];
      QString check = "/";
      if (endvalue != check) {
        outputfileprefix += "_";
      }
    }
    settings->setOutputFilePrefix(outputfileprefix);

    return settings;
  }

  QList<BundleObservationSolveSettings> ceresObservationSolveSettings(UserInterface &ui) {
    //************************************************************************************************
    QList<BundleObservationSolveSettings> observationSolveSettingsList;
    QString fromList = ui.GetFileName("FROMLIST");
    SerialNumberList cubeSNs(fromList);
    // QVector<BundleObservationSolveSettings*> cubeSolveSettings(cubeSNs.size(), nullptr);

    if (ui.WasEntered("SCCONFIG")) {
      PvlObject obj;
      Pvl scConfig(FileName(ui.GetFileName("SCCONFIG")).expanded());
      // QMap<QString, BundleObservationSolveSettings*> instIDtoBOSS;
      if (!scConfig.hasObject("SensorParameters")) {
        QString msg = "Input SCCONFIG file missing SensorParameters object";
        throw IException(IException::User, msg, _FILEINFO_);
      }

      // loop over parameter groups, read settings for each sensor into a
      // BundleObservationSolveSettings object, and append to observationSolveSettingsList
      obj = scConfig.findObject("SensorParameters");
      PvlObject::PvlGroupIterator g;
      for(g = obj.beginGroup(); g != obj.endGroup(); ++g) {
        BundleObservationSolveSettings solveSettings(*g);
        observationSolveSettingsList.append(solveSettings);
      }

      // loop through serial number list, check if current instrumentID matches
      // any of the BOSS's instrumentID. If so, add observationNumber to BOSS.
      for (int snIndex = 0; snIndex < cubeSNs.size(); snIndex++) {
        QString snInstId = cubeSNs.spacecraftInstrumentId(snIndex);
        bool found = false;
        for (auto bossIt = observationSolveSettingsList.begin(); bossIt != observationSolveSettingsList.end(); bossIt++) {
          if (bossIt->instrumentId() == snInstId) {
            bossIt->addObservationNumber(cubeSNs.observationNumber(snIndex));
            found = true;
          }
        }
        if (!found){
          QString msg = "No BundleObservationSolveSettings found for " + snInstId;
          throw IException(IException::User, msg, _FILEINFO_);
        }
      }
    }

    else {
      // We are not using the PVL, so get what will be solve settings for all images from gui
      BundleObservationSolveSettings observationSolveSettings;

      BundleObservationSolveSettings::InstrumentPointingSolveOption pointingSolveOption =
          BundleObservationSolveSettings::stringToInstrumentPointingSolveOption(
              ui.GetString("CAMSOLVE"));

      double anglesAprioriSigma, angularVelocityAprioriSigma, angularAccelerationAprioriSigma;
      anglesAprioriSigma = angularVelocityAprioriSigma = angularAccelerationAprioriSigma = Isis::Null;
      if (ui.WasEntered("CAMERA_ANGLES_SIGMA")) {
        anglesAprioriSigma = ui.GetDouble("CAMERA_ANGLES_SIGMA");
      }
      if (ui.WasEntered("CAMERA_ANGULAR_VELOCITY_SIGMA")) {
        angularVelocityAprioriSigma = ui.GetDouble("CAMERA_ANGULAR_VELOCITY_SIGMA");
      }
      if (ui.WasEntered("CAMERA_ANGULAR_ACCELERATION_SIGMA")) {
        angularAccelerationAprioriSigma = ui.GetDouble("CAMERA_ANGULAR_ACCELERATION_SIGMA");
      }

      observationSolveSettings.setInstrumentPointingSettings(pointingSolveOption,
                                                              ui.GetBoolean("TWIST"),
                                                              ui.GetInteger("CKDEGREE"),
                                                              ui.GetInteger("CKSOLVEDEGREE"),
                                                              ui.GetBoolean("OVEREXISTING"),
                                                              anglesAprioriSigma,
                                                              angularVelocityAprioriSigma,
                                                              angularAccelerationAprioriSigma);

      BundleObservationSolveSettings::InstrumentPositionSolveOption positionSolveOption =
          BundleObservationSolveSettings::stringToInstrumentPositionSolveOption(
              ui.GetString("SPSOLVE"));

      double positionAprioriSigma, positionVelocityAprioriSigma, positionAccelerationAprioriSigma;
      positionAprioriSigma = positionVelocityAprioriSigma = positionAccelerationAprioriSigma
                            = Isis::Null;
      if ( ui.WasEntered("SPACECRAFT_POSITION_SIGMA") ) {
        positionAprioriSigma = ui.GetDouble("SPACECRAFT_POSITION_SIGMA");
      }
      if ( ui.WasEntered("SPACECRAFT_VELOCITY_SIGMA") ) {
        positionVelocityAprioriSigma = ui.GetDouble("SPACECRAFT_VELOCITY_SIGMA");
      }
      if ( ui.WasEntered("SPACECRAFT_ACCELERATION_SIGMA") ) {
        positionAccelerationAprioriSigma = ui.GetDouble("SPACECRAFT_ACCELERATION_SIGMA");
      }

      observationSolveSettings.setInstrumentPositionSettings(positionSolveOption,
                                                              ui.GetInteger("SPKDEGREE"),
                                                              ui.GetInteger("SPKSOLVEDEGREE"),
                                                              ui.GetBoolean("OVERHERMITE"),
                                                              positionAprioriSigma,
                                                              positionVelocityAprioriSigma,
                                                              positionAccelerationAprioriSigma);

      if ((ui.WasEntered("CSMSOLVESET")  && ui.WasEntered("CSMSOLVETYPE")) ||
          (ui.WasEntered("CSMSOLVESET")  && ui.WasEntered("CSMSOLVELIST")) ||
          (ui.WasEntered("CSMSOLVETYPE") && ui.WasEntered("CSMSOLVELIST")) ) {
        QString msg = "Only one of CSMSOLVESET, CSMSOLVETYPE, and CSMSOLVELIST "
                      "can be specified at a time.";
        throw IException(IException::User, msg, _FILEINFO_);
      }

      if (ui.WasEntered("CSMSOLVESET")) {
        observationSolveSettings.setCSMSolveSet(
            BundleObservationSolveSettings::stringToCSMSolveSet(ui.GetString("CSMSOLVESET")));
      }
      else if (ui.WasEntered("CSMSOLVETYPE")) {
        observationSolveSettings.setCSMSolveType(
            BundleObservationSolveSettings::stringToCSMSolveType(ui.GetString("CSMSOLVETYPE")));
      }
      else if (ui.WasEntered("CSMSOLVELIST")) {
        std::vector<QString> csmParamVector;
        ui.GetString("CSMSOLVELIST", csmParamVector);
        QStringList csmParamList = QStringList::fromVector(QVector<QString>(csmParamVector.begin(), csmParamVector.end()));
        observationSolveSettings.setCSMSolveParameterList(csmParamList);
      }

      // add all image observation numbers to this BOSS.
      for (int sn = 0; sn < cubeSNs.size(); sn++) {
        observationSolveSettings.addObservationNumber(cubeSNs.observationNumber(sn));
      }

      // append the GUI acquired solve parameters to BOSS list.
      observationSolveSettingsList.append(observationSolveSettings);

    }


    // If we are holding any images, then we need a BundleObservationSolveSettings for the held
    // images, and another one for the non-held images
    if (ui.WasEntered("HELDLIST")) {
      // Check that the held images are present in the input image list
      QString heldList = ui.GetFileName("HELDLIST");
      SerialNumberList heldSNs(heldList);
      ceresCheckImageList(heldSNs, cubeSNs);

      double anglesAprioriSigma, angularVelocityAprioriSigma, angularAccelerationAprioriSigma;
      anglesAprioriSigma = angularVelocityAprioriSigma = angularAccelerationAprioriSigma = Isis::Null;

      double positionAprioriSigma, positionVelocityAprioriSigma, positionAccelerationAprioriSigma;
      positionAprioriSigma = positionVelocityAprioriSigma = positionAccelerationAprioriSigma
                            = Isis::Null;

      // The settings for the held images will have no pointing or position factors considered
      BundleObservationSolveSettings heldSettings;
      BundleObservationSolveSettings::InstrumentPointingSolveOption noPointing =
          BundleObservationSolveSettings::stringToInstrumentPointingSolveOption(
              "NoPointingFactors");
      heldSettings.setInstrumentPointingSettings(noPointing,
                                                  ui.GetBoolean("TWIST"),
                                                  ui.GetInteger("CKDEGREE"),
                                                  ui.GetInteger("CKSOLVEDEGREE"),
                                                  ui.GetBoolean("OVEREXISTING"),
                                                  anglesAprioriSigma,
                                                  angularVelocityAprioriSigma,
                                                  angularAccelerationAprioriSigma);
      BundleObservationSolveSettings::InstrumentPositionSolveOption noPosition =
          BundleObservationSolveSettings::stringToInstrumentPositionSolveOption(
              "NoPositionFactors");
      heldSettings.setInstrumentPositionSettings(noPosition,
                                                  ui.GetInteger("SPKDEGREE"),
                                                  ui.GetInteger("SPKSOLVEDEGREE"),
                                                  ui.GetBoolean("OVERHERMITE"),
                                                  positionAprioriSigma,
                                                  positionVelocityAprioriSigma,
                                                  positionAccelerationAprioriSigma);

      // Add the held images' observationNumbers to the held observation solve settings
      for (int sn = 0; sn < cubeSNs.size(); sn++) {
        if ( heldSNs.hasSerialNumber(cubeSNs.serialNumber(sn)) ) {
          // For held images, we want to set pointing and position settings to NONE, effectively
          // ensuring that the number of pointing and position parameters for the holds are 0
          heldSettings.addObservationNumber(cubeSNs.observationNumber(sn));
          QString snInstId = cubeSNs.spacecraftInstrumentId(sn);
          //for each held serial number, locate corresponding BOSS in BOSSlist, remove.
          for (auto bossIt = observationSolveSettingsList.begin(); bossIt != observationSolveSettingsList.end(); bossIt++) {
            if (bossIt->instrumentId() == snInstId) {
              bossIt->removeObservationNumber(cubeSNs.observationNumber(sn));
            }
          }
        }
      }
      // Add the held observation solve settings to the list of solve settings
      // for the BundleAdjust
      observationSolveSettingsList.append(heldSettings);
    }

    //************************************************************************************************
    return observationSolveSettingsList;
  }

  /**
   * Checks that all the first serial numbers are in the second serial numbers list (FROMLIST).
   *
   * Note: This function is used for verifying HELDLIST is in the FROMLIST.
   *
   * @param imageList List of image serial numbers to check to make sure they're in the second list.
   * @param fromList List of input image serial numbers to check against.
   *
   * @throws IException::User "The following images are not in the FROMLIST:"
   */
  void ceresCheckImageList(SerialNumberList &imageList, SerialNumberList &fromList) {
    // Keep track of which held images are not in the FROMLIST
    QString imagesNotFound;

    for (int img = 0; img < imageList.size(); img++) {
      // When the FROMLIST does not have the current image, record the image's filename
      if ( !fromList.hasSerialNumber(imageList.serialNumber(img)) ) {
        imagesNotFound += " [" + imageList.fileName(img) + "]";
      }
    }

    // Inform the user which images are not in the second list
    if (!imagesNotFound.isEmpty()) {
      QString msg = "The following images are not in the FROMLIST:";
      msg += imagesNotFound + ".";
      throw IException(IException::User, msg, _FILEINFO_);
    }
  }

  /**
   * Compute partial derivatives and weighted residuals for a measure.
   * coeffTarget, coeffImage, coeffPoint3D, and coeffRHS will be filled
   * with the different partial derivatives.
   *
   * @param coeffTarget A matrix that will contain target body
   *                    partial derivatives.
   * @param coeffImage A matrix that will contain camera position and orientation
   *                   partial derivatives.
   * @param coeffPoint3D A matrix that will contain point lat, lon, and radius
   *                     partial derivatives.
   * @param coeffRHS A vector that will contain weighted x,y residuals.
   * @param measure The measure that partials are being computed for.
   * @param point The point containing measure.
   *
   * @return @b bool If the partials were successfully computed.
   *
   * @throws IException::User "Unable to map apriori surface point for measure"
   */
  bool computePartials(MeasurePartials &partials,
                       BundleMeasure &measure,
                       BundleControlPoint &point,
                       QSharedPointer<BundleSettings> settings,
                       BundleResults &bundleResults) {

    LinearAlgebra::Matrix &coeffTarget = partials.coeffTarget;
    LinearAlgebra::Matrix &coeffImage = partials.coeffImage;
    LinearAlgebra::Matrix &coeffPoint3D = partials.coeffPoint3D;
    LinearAlgebra::Vector &coeffRHS = partials.coeffRHS;

    Camera *measureCamera = measure.camera();
    BundleObservationQsp observation = measure.parentBundleObservation();
    const SurfacePoint &adjustedSurfacePoint = point.adjustedSurfacePoint();

    int numImagePartials = observation->numberParameters();

    // only resize when this observation's parameter count differs from the last fill
    if ((int) coeffImage.size2() != numImagePartials) {
      coeffImage.resize(2,numImagePartials);
    }

    // No need to call SetImage for framing camera
    if (measureCamera->GetCameraType() != Camera::Framing) {
      // Set the Spice to the measured point.  A framing camera exposes the entire image at one time.
      // It will have a single set of Spice for the entire image.  Scanning cameras may populate a single
      // image with multiple exposures, each with a unique set of Spice.  SetImage needs to be called
      // repeatedly for these images to point to the Spice for the current pixel.
      measureCamera->SetImage(measure.sample(), measure.line());
    }

    // CSM Cameras do not have a ground map
    if (measureCamera->GetCameraType() != Camera::Csm) {
      // Compute the look vector in instrument coordinates based on time of observation and apriori
      // lat/lon/radius.  As of 05/15/2019, this call no longer does the back-of-planet test. An optional
      // bool argument was added CameraGroundMap::GetXY to turn off the test.
      double computedX, computedY;
      if (!(measureCamera->GroundMap()->GetXY(adjustedSurfacePoint,
                                              &computedX, &computedY, false))) {
        QString msg = "Unable to map apriori surface point for measure ";
        msg += measure.cubeSerialNumber() + " on point " + point.id() + " into focal plane";
        throw IException(IException::User, msg, _FILEINFO_);
      }

      // this is what ComputeResiduals_Millimeters would compute for the same parameters
      measure.setFocalPlaneComputed(computedX, computedY);
    }
    // if (settings->solveTargetBody()) {
    //   observation->computeTargetPartials(coeffTarget, measure, settings, m_bundleTargetBody);
    // }

    observation->computeImagePartials(coeffImage, measure);

    // Complete partials calculations for 3D point (latitudinal or rectangular)
    // Retrieve the coordinate type (latitudinal or rectangular) and compute the partials for
    // the fixed point with respect to each coordinate in Body-Fixed
    SurfacePoint::CoordinateType coordType = settings->controlPointCoordTypeBundle();
    observation->computePoint3DPartials(coeffPoint3D, measure, adjustedSurfacePoint, coordType);

    // right-hand side (measured - computed)
    observation->computeRHSPartials(coeffRHS, measure);

    double deltaX = coeffRHS(0);
    double deltaY = coeffRHS(1);

    // the running distributions are order dependent, so record the observations here and add
    // them from the thread that owns m_bundleResults
    partials.residualX = observation->computeObservationValue(measure, deltaX);
    partials.residualY = observation->computeObservationValue(measure, deltaY);

    if (bundleResults.numberMaximumLikelihoodModels()
          > bundleResults.maximumLikelihoodModelIndex()) {
      // If maximum likelihood estimation is being used
      double residualR2ZScore = sqrt(deltaX * deltaX + deltaY * deltaY) / sqrt(2.0);
      partials.residualR2ZScore = residualR2ZScore;

      int currentModelIndex = bundleResults.maximumLikelihoodModelIndex();
      double observationWeight = bundleResults.maximumLikelihoodModelWFunc(currentModelIndex)
                            .sqrtWeightScaler(residualR2ZScore);
      coeffImage *= observationWeight;
      coeffPoint3D *= observationWeight;
      coeffRHS *= observationWeight;

      // if (settings->solveTargetBody()) {
      //   coeffTarget *= observationWeight;
      // }
    }

    partials.observationIndex = measure.observationIndex();

    return true;
  }
}

