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
#include "ControlMeasure.h"
#include "ControlNet.h"
#include "ControlPoint.h"
#include "CubeAttribute.h"
#include "Displacement.h"
#include "IException.h"
#include "iTime.h"
#include "MaximumLikelihoodWFunctions.h"
#include "Process.h"
// #include "SensorUtilities.h"
#include "SerialNumber.h"
#include "SerialNumberList.h"
#include "Table.h"
#include "CameraFactory.h"
#include "PvlToJSON.h"
#include "PvlKeyword.h"

#include <ale/Load.h>
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
using json = nlohmann::json;

#include <ceres/ceres.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/cost_function_to_functor.h>
#include <ceres/autodiff_cost_function.h>

#include "ceres_jigsaw.h"

using namespace std;
using namespace HighFive;

namespace Isis {
  BundleSettingsQsp ceresBundleSettings(UserInterface &ui);
  void ceresCheckImageList(SerialNumberList &heldSerialList, SerialNumberList &cubeSerialList);
  QList<BundleObservationSolveSettings> ceresObservationSolveSettings(UserInterface &ui);

  // 9 potential position coefficients
  // 9 potentail rotation coefficients
  // const int numParams = 9 + 9;
  const int numParams = 9 + 9;

  BundleObservationSolveSettings::InstrumentPositionSolveOption solvePosition = BundleObservationSolveSettings::InstrumentPositionSolveOption::NoPositionFactors;
  BundleObservationSolveSettings::InstrumentPointingSolveOption solveRotation = BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors;

  struct SnavelyReprojectionFunctor {
    SnavelyReprojectionFunctor(Camera *camera) : camera(camera) {
      camera->instrumentPosition()->GetPolynomial(originalPositionPoly1,
                                                  originalPositionPoly2,
                                                  originalPositionPoly3);
      camera->instrumentRotation()->GetPolynomial(originalPointingPoly1,
                                                  originalPointingPoly2,
                                                  originalPointingPoly3);
    }

    bool operator()(const double* calibration, const double *point, double* predicted) const {
      // Cube cube(camera_file);
      // Camera *camera = CameraFactory::Create(cube);
      SpicePosition *instPosition = camera->instrumentPosition();
      SpiceRotation *instPointing = camera->instrumentRotation();

      std::vector<double> poly1(3), poly2(3), poly3(3);
      poly1 = originalPositionPoly1;
      poly2 = originalPositionPoly2;
      poly3 = originalPositionPoly3;
      if (solvePosition != BundleObservationSolveSettings::InstrumentPositionSolveOption::NoPositionFactors) {
        // Based on solve settings we need to only update the correct coeffs
        // m_instrumentPositionSolveOption
        // NONE - no updates
        // POSITIONS - 0th element in each polynomail
        // VELOCITEIS - 0th and 1st elements in each polynomail
        // ACCELERATIONS - 0th, 1st and 2nd elements in each polynomail
        // ALL - 0th, 1st and 2nd elements in each polynomail
        std::copy(&calibration[0], &(calibration[0 + (int)solvePosition]), poly1.begin());
        std::copy(&calibration[3], &(calibration[3 + (int)solvePosition]), poly2.begin());
        std::copy(&calibration[6], &(calibration[6 + (int)solvePosition]), poly3.begin());
      }
      instPosition->SetPolynomial(poly1, poly2, poly3);

      poly1 = originalPointingPoly1;
      poly2 = originalPointingPoly2;
      poly3 = originalPointingPoly3;
      if (solveRotation != BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors) {
        // Based on solve settings we need to only update the correct coeffs
        // m_instrumentPointingSolveOption
        // NONE - no updates
        // ANGLES - 0th element in each polynomail
        // VELOCITEIS - 0th and 1st elements in each polynomail
        // ACCELERATIONS - 0th, 1st and 2nd elements in each polynomail
        // ALL - 0th, 1st and 2nd elements in each polynomail
        std::copy(&calibration[9], &(calibration[9 + (int)solveRotation]), poly1.begin());
        std::copy(&calibration[12], &(calibration[12 + (int)solveRotation]), poly2.begin());
        std::copy(&calibration[15], &(calibration[15 + (int)solveRotation]), poly3.begin());
      }
      instPointing->SetPolynomial(poly1, poly2, poly3);
      
      Displacement x(point[0], Displacement::Units::Meters);
      Displacement y(point[1], Displacement::Units::Meters);
      Displacement z(point[2], Displacement::Units::Meters);

      SurfacePoint surfacePoint(x, y, z);
      if (!camera->SetGround(surfacePoint)) {
        // Return false if point is not visible
        return false;
      }

      predicted[0] = camera->Sample();
      predicted[1] = camera->Line();

      return true;
    }

    Camera *camera;
    std::vector<double> originalPositionPoly1, originalPositionPoly2, originalPositionPoly3;
    std::vector<double> originalPointingPoly1, originalPointingPoly2, originalPointingPoly3;
  };


  struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y, Camera *camera)
        : observed_x(observed_x), observed_y(observed_y) {

      ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<SnavelyReprojectionFunctor, ceres::CENTRAL, 2, numParams, 3>
            (new SnavelyReprojectionFunctor(camera));

      compute_point = std::make_unique<ceres::CostFunctionToFunctor<2, numParams, 3>>(cost_function);
    }

    template <typename T>
    bool operator()(const T* calibration, const T *point, T* residuals) const {
      T predicted[2];
      (*compute_point)(calibration, point, predicted);
      residuals[0] = observed_x - predicted[0];
      residuals[1] = observed_y - predicted[1];
      return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       Camera *camera) {
      return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, numParams, 3>
        (new SnavelyReprojectionError(observed_x, observed_y, camera));
    }

    double observed_x;
    double observed_y;
    std::unique_ptr<ceres::CostFunctionToFunctor<2, numParams, 3>> compute_point;
  };

  void ceres_jigsaw(UserInterface &ui, Pvl *log) {
    Progress progress;
    BundleSettingsQsp settings = ceresBundleSettings(ui);

    // initialize solution parameters
    BundleObservationSolveSettings solveSettings = settings->observationSolveSettings(0);
    BundleObservationSolveSettings::InstrumentPointingSolveOption pointingSolveOption = BundleObservationSolveSettings::stringToInstrumentPointingSolveOption(ui.GetString("CAMSOLVE"));

    ControlNet network(ui.GetFileName("CNET"));
    SerialNumberList snList(ui.GetFileName("FROMLIST"));
    network.SetImages(snList, &progress);
    std::map<QString, double *> polyMap;
    for (int i = 0; i < snList.size(); i++) {
      QString serialNumber = snList.serialNumber(i);
      double *cameraPolynomials = new double[numParams];
      Camera *camera = network.Camera(serialNumber);

      SpicePosition *spicePosition = camera->instrumentPosition();
      
      // first, set the degree of the spk polynomial to be fit for a priori values
      spicePosition->SetPolynomialDegree(solveSettings.spkDegree());

      // now, set what kind of interpolation to use (polynomial function or
      // polynomial function over hermite spline)
      spicePosition->SetPolynomial(solveSettings.positionInterpolationType());

      // finally, set the degree of the position polynomial actually used in the bundle adjustment
      spicePosition->SetPolynomialDegree(solveSettings.spkSolveDegree());
      std::vector<double> positionPoly1, positionPoly2, positionPoly3;
      spicePosition->GetPolynomial(positionPoly1, positionPoly2, positionPoly3);
      copy(positionPoly1.begin(), positionPoly1.end(), &cameraPolynomials[0]);
      copy(positionPoly2.begin(), positionPoly2.end(), &cameraPolynomials[3]);
      copy(positionPoly3.begin(), positionPoly3.end(), &cameraPolynomials[6]);

      SpiceRotation *spiceRotation = camera->instrumentRotation();

      // first, set the degree of the polynomial to be fit for a priori values
      spiceRotation->SetPolynomialDegree(solveSettings.ckDegree());

      // now, set what kind of interpolation to use (polynomial function or
      // polynomial function over a pointing cache)
      spiceRotation->SetPolynomial(solveSettings.pointingInterpolationType());

      // finally, set the degree of the pointing polynomial actually used in the bundle adjustment
      spiceRotation->SetPolynomialDegree(solveSettings.ckSolveDegree());
      std::vector<double> anglePoly1, anglePoly2, anglePoly3;
      spiceRotation->GetPolynomial(anglePoly1, anglePoly2, anglePoly3);
      copy(anglePoly1.begin(), anglePoly1.end(), &cameraPolynomials[9]);
      copy(anglePoly2.begin(), anglePoly2.end(), &cameraPolynomials[12]);
      copy(anglePoly3.begin(), anglePoly3.end(), &cameraPolynomials[15]);
      polyMap[serialNumber] = cameraPolynomials;
      std::cout << polyMap[serialNumber] << std::endl;
    }
    // Convert to pointer of similar data
    // That is, each entry should point to some cameras set of
    // polys rather than making a duplicate entry
    double **polynomials = new double*[network.GetNumValidMeasures()];
    double **observedPoints = new double*[network.GetNumValidMeasures()];
    double **groundPoints = new double*[network.GetNumValidMeasures()];
    for (int i = 0; i < network.GetNumValidMeasures(); i++) {
      observedPoints[i] = new double[2];
      for (int j = 0; j < 2; j++) {
        observedPoints[i][j] = 0;
      }
      groundPoints[i] = new double[3];
      for (int j = 0; j < 3; j++) {
        groundPoints[i][j] = 0;
      }
    }

    std::vector<Camera *> cameras(network.GetNumValidMeasures());
    int vector_idx = 0;
    for (int i = 0; i < network.GetNumPoints(); i++) {
      ControlPoint *point = network.GetPoint(i);
      if (point->IsIgnored()) {
        continue;
      }
      ControlMeasure *refMeasure = point->GetRefMeasure();
      refMeasure->Camera()->SetImage(refMeasure->GetSample(), refMeasure->GetLine());
      double groundCoord[3];
      refMeasure->Camera()->Coordinate(groundCoord);
      for (int j = 0; j < point->GetNumMeasures(); j++) {
        ControlMeasure *measure = point->GetMeasure(j);
        if (measure->IsIgnored()) {
          continue;
        }
        observedPoints[vector_idx][0] = measure->GetSample();
        observedPoints[vector_idx][1] = measure->GetLine();

        QString serialNumber = measure->GetCubeSerialNumber();
        cameras[vector_idx] = network.Camera(serialNumber);
        std::copy(&groundCoord[0], &groundCoord[3], &groundPoints[vector_idx][0]);
        groundPoints[vector_idx][0] *= 1000;
        groundPoints[vector_idx][1] *= 1000;
        groundPoints[vector_idx][2] *= 1000;
        polynomials[vector_idx] = polyMap[serialNumber];
        std::cout << snList.fileName(serialNumber) << std::endl;
        std::cout << cameras[vector_idx] << std::endl;
        std::cout << polynomials[vector_idx] << std::endl;
        vector_idx++;
      }
    }

    ceres::Problem problem;
    for (int i = 0; i < network.GetNumValidMeasures(); ++i) {
      ceres::CostFunction* cost_function =
          SnavelyReprojectionError::Create(observedPoints[i][0],
                                           observedPoints[i][1],
                                           cameras[i]);
      problem.AddResidualBlock(cost_function,
                              nullptr /* squared loss */,
                              polynomials[i],
                              groundPoints[i]);
    }

    solvePosition = solveSettings.instrumentPositionSolveOption();
    solveRotation = solveSettings.instrumentPointingSolveOption();
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << std::setprecision(15);
    for (int i = 0; i < snList.size(); i++) {
      QString serialNumber = snList.serialNumber(i);
      QString fileName = snList.fileName(i);
      Cube cube(snList.fileName(i));
      Camera *originalCamera = cube.camera();
      Camera *networkCamera = network.Camera(serialNumber);
      // compare Angles of original camera against new angles
      const std::vector<double> networkCoord = networkCamera->instrumentPosition()->GetCenterCoordinate();
      const std::vector<double> networkAngles = networkCamera->instrumentRotation()->GetCenterAngles();

      const std::vector<double> originalCoord = originalCamera->instrumentPosition()->GetCenterCoordinate();
      const std::vector<double> originalAngles = originalCamera->instrumentRotation()->GetCenterAngles();
      std::cout << snList.fileName(serialNumber) << std::endl;

      std::cout << networkCoord[0] - originalCoord[0] << ", ";
      std::cout << networkCoord[1] - originalCoord[1] << ", ";
      std::cout << networkCoord[2] - originalCoord[2] << std::endl << std::endl;

      std::cout << networkAngles[0] - originalAngles[0] << ", ";
      std::cout << networkAngles[1] - originalAngles[1] << ", ";
      std::cout << networkAngles[2] - originalAngles[2] << std::endl << std::endl;
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
          std::vector<double> positionPoly1(3), positionPoly2(3), positionPoly3(3);
          copy(&updatedCameraPolynomials[0], &updatedCameraPolynomials[0 + 3], positionPoly1.begin());
          copy(&updatedCameraPolynomials[3], &updatedCameraPolynomials[3 + 3], positionPoly2.begin());
          copy(&updatedCameraPolynomials[6], &updatedCameraPolynomials[6 + 3], positionPoly3.begin());
          camera->instrumentPosition()->SetPolynomial(positionPoly1, positionPoly2, positionPoly3);
          Table spvector = camera->instrumentPosition()->Cache("InstrumentPosition");
          spvector.Label().addComment(jigComment);
          cube->write(spvector);
        }

        if (solveRotation != BundleObservationSolveSettings::InstrumentPointingSolveOption::NoPointingFactors) {
          std::vector<double> anglePoly1(3), anglePoly2(3), anglePoly3(3);
          copy(&updatedCameraPolynomials[9], &updatedCameraPolynomials[9 + 3], anglePoly1.begin());
          copy(&updatedCameraPolynomials[12], &updatedCameraPolynomials[12 + 3], anglePoly2.begin());
          copy(&updatedCameraPolynomials[15], &updatedCameraPolynomials[15 + 3], anglePoly3.begin());
          camera->instrumentRotation()->SetPolynomial(anglePoly1, anglePoly2, anglePoly3);
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

    for (int i = 0; i < network.GetNumValidMeasures(); i++) {
      delete []observedPoints[i];
      delete []groundPoints[i];
    }

    delete []polynomials;
    delete []observedPoints;
    delete []groundPoints;
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

    double coord1Sigma  = Isis::Null;
    double coord2Sigma = Isis::Null;
    double coord3Sigma    = Isis::Null;
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
}
