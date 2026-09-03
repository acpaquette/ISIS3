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
  BundleSettingsQsp ceresBundleSettings(UserInterface &ui);
  void ceresCheckImageList(SerialNumberList &heldSerialList, SerialNumberList &cubeSerialList);
  QList<BundleObservationSolveSettings> ceresObservationSolveSettings(UserInterface &ui);

  // 9 potential position coefficients
  // 9 potentail rotation coefficients
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

  void ceres_jigsaw(UserInterface &ui, Pvl *log) {
    Progress progress;
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

    ControlNet network(ui.GetFileName("CNET"));
    SerialNumberList snList(ui.GetFileName("FROMLIST"));
    network.SetImages(snList, &progress);
    std::map<QString, double *> polyMap;
    std::map<QString, double *> originalPolyMap;
    for (int i = 0; i < snList.size(); i++) {
      QString serialNumber = snList.serialNumber(i);
      std::cout << serialNumber << std::endl;
      double *cameraPolynomials = new double[numParams];
      double *originalCameraPolynomials = new double[numParams];
      for (int j = 0; j < numParams; j++) {
        cameraPolynomials[j] = 0.0;
        originalCameraPolynomials[j] = 0.0;
      }
      Camera *camera = network.Camera(serialNumber);
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

      for (int j = 0; j < numParams; j++) {
        std::cout << cameraPolynomials[j] << std::endl;
      }
      std::cout << std::endl;
      polyMap[serialNumber] = cameraPolynomials;
      originalPolyMap[serialNumber] = originalCameraPolynomials;
    }

    // Convert to pointer of similar data
    // That is, each entry should point to some cameras set of
    // polys rather than making a duplicate entry
    std::vector<PointParameters> bundlePointParameters(network.GetNumValidMeasures());
    std::vector<GroundParameters> bundleGroundParameters(network.GetNumValidPoints());
    double **observedPoints = new double*[network.GetNumValidMeasures()];

    std::vector<Camera *> cameras(network.GetNumValidMeasures(), nullptr);
    std::vector<double> measureSigmas(network.GetNumValidMeasures(), 1.0);
    std::vector<std::vector<double>> observedGround(network.GetNumValidPoints(), std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> groundSigmas(network.GetNumValidPoints(), std::vector<double>(3, 1.0));
    int vector_idx = 0;
    progress.SetText("Loading ceres data...");
    progress.SetMaximumSteps(network.GetNumPoints());
    progress.CheckStatus();
    for (int i = 0; i < network.GetNumPoints(); i++) {
      if (network.GetPoint(i)->IsIgnored()) {
        continue;
      }
      BundleControlPoint point = BundleControlPoint(settings, network.GetPoint(i));
      point.rawControlPoint()->ComputeApriori();
      
      SurfacePoint ground = point.rawControlPoint()->GetAprioriSurfacePoint();
      double *groundCoord = new double[3];
      groundCoord[0] = ground.GetX().kilometers();
      groundCoord[1] = ground.GetY().kilometers();
      groundCoord[2] = ground.GetZ().kilometers();
      
      bundleGroundParameters[i] = GroundParameters(groundCoord);
      observedGround[i] = {groundCoord[0], groundCoord[1], groundCoord[2]};
      groundSigmas[i] = std::vector<double>(point.aprioriSigmas().begin(), point.aprioriSigmas().end());
      for (int j = 0; j < point.numberOfMeasures(); j++) {
        QSharedPointer<BundleMeasure> measure = point.at(j);

        observedPoints[vector_idx] = new double[2];
        observedPoints[vector_idx][0] = measure->sample();
        observedPoints[vector_idx][1] = measure->line();

        QString serialNumber = measure->cubeSerialNumber();
        cameras[vector_idx] = measure->camera();
        
        // Magic number from ISIS implementation. This does not actually set the sigma to 1.4
        // See function for more detail
        measure->setSigma(1.4);
        // Do we want weight or sigma?
        // try with sigma, see what happens
        measureSigmas[vector_idx] = measure->sigma();

        // Copy data to spots in pointParameters
        bundlePointParameters[vector_idx] = PointParameters(polyMap[serialNumber], groundCoord, rotationParamSize, positionParamSize);
        vector_idx++;
      }
      progress.CheckStatus();
    }

    std::cout << std::setprecision(15);
    for ( int i = 0; i < 10; i++) {
      std::cout << "INITIAL GP: ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][0] << ", ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][1] << ", ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][2] << std::endl;
    }

    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    ceres::Problem problem;
    for (int i = 0; i < network.GetNumValidMeasures(); ++i) {
      auto* cost_function =
          SnavelyReprojectionErrorFunctor::Create(observedPoints[i][0],
                                                  observedPoints[i][1],
                                                  cameras[i],
                                                  positionParamSize,
                                                  rotationParamSize,
                                                  measureSigmas[i],
                                                  &solveSettings);
      for (int j = 0; j < positionParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      for (int j = 0; j < rotationParamSize; j++) {
        cost_function->AddParameterBlock(3);
      }
      cost_function->AddParameterBlock(3);
      cost_function->SetNumResiduals(2);
      problem.AddResidualBlock(cost_function,
                               loss_function,
                               bundlePointParameters[i].parameters);
      if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocityAcceleration) {
        problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[2]);
        if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionVelocity) {
          problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[1]);
          if (solvePosition < BundleObservationSolveSettings::InstrumentPositionSolveOption::PositionOnly) {
            problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[0]);
          }
        }
      }

      if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocityAcceleration) {
        problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[5]);
        if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesVelocity) {
          problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[4]);
          if (solveRotation < BundleObservationSolveSettings::InstrumentPointingSolveOption::AnglesOnly) {
            problem.SetParameterBlockConstant(bundlePointParameters[i].parameters[3]);
          }
        }
      }
    }

    for (int i = 0; i < network.GetNumValidPoints(); i++) {
      auto* cost_function = XYZError::Create(observedGround[i], groundSigmas[i]);
      ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
      problem.AddResidualBlock(cost_function, loss_function, bundleGroundParameters[i].parameters);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.parameter_tolerance = ui.GetDouble("SIGMA0");
    options.max_num_iterations = ui.GetInteger("MAXITS");
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    for ( int i = 0; i < 10; i++) {
      std::cout << "POST GP: ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][0] << ", ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][1] << ", ";
      std::cout << bundlePointParameters[i].parameters[rotationParamSize + positionParamSize][2] << std::endl;
    }
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
    for (int i = 0; i < network.GetNumPoints(); i++) {
      ControlPoint *point = network.GetPoint(i);
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
