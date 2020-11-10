#include <QTextStream>

#include "Fixtures.h"
#include "LineManager.h"
#include "SpecialPixel.h"
#include "ControlNet.h"

namespace Isis {

  void TempTestingFiles::SetUp() {
    ASSERT_TRUE(tempDir.isValid());
  }


  void SmallCube::SetUp() {
    TempTestingFiles::SetUp();

    testCube = new Cube();
    testCube->setDimensions(10, 10, 10);
    testCube->create(tempDir.path() + "/small.cub");

    LineManager line(*testCube);
    double pixelValue = 0.0;
    for(line.begin(); !line.end(); line++) {
      for(int i = 0; i < line.size(); i++) {
        line[i] = (double) pixelValue++;
      }
      testCube->write(line);
    }
  }

  void SmallCube::TearDown() {
    if (testCube->isOpen()) {
      testCube->close();
    }

    if (testCube) {
      delete testCube;
    }
  }


  void LargeCube::SetUp() {
    TempTestingFiles::SetUp();

    testCube = new Cube();
    testCube->setDimensions(1000, 1000, 10);
    testCube->create(tempDir.path() + "/large.cub");

    LineManager line(*testCube);
    double pixelValue = 0.0;
    for(line.begin(); !line.end(); line++) {
      for(int i = 0; i < line.size(); i++) {
        line[i] = pixelValue;
      }

      pixelValue++;
      testCube->write(line);
    }
  }

  void LargeCube::TearDown() {
    if (testCube->isOpen()) {
      testCube->close();
    }

    if (testCube) {
      delete testCube;
    }
  }


  void SpecialSmallCube::SetUp() {
    TempTestingFiles::SetUp();

    testCube = new Cube();
    testCube->setDimensions(10, 10, 10);
    testCube->create(tempDir.path() + "/small.cub");

    // Use a line manager to update select lines with ISIS special pixel values
    LineManager line(*testCube);
    double pixelValue = 0.0;
    int lineNum = 0;
    for(line.begin(); !line.end(); line++) {
      for(int i = 0; i < line.size(); i++) {
        if (lineNum == 2) {
          line[i] = NULL8;
        }
        else if (lineNum == 3) {
          line[i] = LOW_REPR_SAT8;
        }
        else if (lineNum == 4) {
          line[i] = HIGH_REPR_SAT8;
        }
        else if (lineNum == 5) {
          line[i] = LOW_INSTR_SAT8;
        }
        else if (lineNum == 6) {
          line[i] = HIGH_INSTR_SAT8;
        }
        else {
          line[i] = (double) pixelValue++;
        }
      }
      lineNum++;
      testCube->write(line);
    }

  }

  void SpecialSmallCube::TearDown() {
    if (testCube->isOpen()) {
      testCube->close();
    }

    if (testCube) {
      delete testCube;
    }
  }

  void DemCube::SetUp() {
    DefaultCube::SetUp();

    std::ifstream cubeLabel("data/defaultImage/mola.pvl");

    Pvl demLabel;
    cubeLabel >> demLabel;

    testCube->label()->object(4)["SolarLongitude"] = "294.73518831328";
    testCube->reopen("rw");

    demCube = new Cube();
    demCube->fromLabel("/Users/acpaquette/Desktop/demCube.cub", demLabel, "rw");

    TableField minRadius("MinimumRadius", TableField::Double);
    TableField maxRadius("MaximumRadius", TableField::Double);

    TableRecord record;
    record += minRadius;
    record += maxRadius;

    Table shapeModelStatistics("ShapeModelStatistics", record);

    record[0] = 3376.2;
    record[1] = 3396.19;
    shapeModelStatistics += record;

    demCube->write(shapeModelStatistics);

    int xCenter = int(demCube->lineCount()/2);
    int yCenter = int(demCube->sampleCount()/2);
    double radius = std::min(xCenter, yCenter);
    double depth = 30;
    double pointRadius;

    LineManager line(*demCube);
    double pixelValue;
    double base = demCube->label()->findObject("IsisCube").findObject("Core").findGroup("Pixels")["Base"];
    double xPos = 0.0;

    for(line.begin(); !line.end(); line++) {
      for(int yPos = 0; yPos < line.size(); yPos++) {
        pointRadius = pow(pow((xPos - xCenter), 2) + pow((yPos - yCenter), 2), 0.5);
        if (pointRadius < radius) {
          pixelValue = ((sin(((M_PI*pointRadius)/(2*radius))) * depth) + depth) + base;
        }
        else {
          pixelValue = base + (depth * 2);
        }
        line[yPos] = (double) pixelValue;
      }
      xPos++;
      demCube->write(line);
    }

    demCube->reopen("rw");
  }

  void DemCube::TearDown() {
    if (demCube->isOpen()) {
      demCube->close();
    }

    delete demCube;
  }


  void DefaultCube::SetUp() {
    TempTestingFiles::SetUp();

    std::ifstream isdFile("data/defaultImage/defaultCube.isd");
    std::ifstream cubeLabel("data/defaultImage/defaultCube.pvl");
    std::ifstream projCubeLabel("data/defaultImage/projDefaultCube.pvl");

    isdFile >> isd;
    cubeLabel >> label;
    projCubeLabel >> projLabel;

    testCube = new Cube();
    testCube->fromIsd(tempDir.path() + "/default.cub", label, isd, "rw");

    projTestCube = new Cube();
    projTestCube->fromIsd(tempDir.path() + "/default.level2.cub", projLabel, isd, "rw");
  }


  void DefaultCube::TearDown() {
    if (testCube->isOpen()) {
      testCube->close();
    }
    
    if (projTestCube->isOpen()) {
      projTestCube->close();
    }
    
    delete testCube;
    delete projTestCube;
  }

  void LineScannerCube::SetUp() {
    TempTestingFiles::SetUp();

    std::ifstream isdFile("data/LineScannerImage/defaultLineScanner.isd");
    std::ifstream cubeLabel("data/LineScannerImage/defaultLineScanner.pvl");
    std::ifstream projCubeLabel("data/LineScannerImage/projDefaultLineScanner.pvl");

    isdFile >> isd;
    cubeLabel >> label;
    projCubeLabel >> projLabel;

    testCube = new Cube();
    testCube->fromIsd(tempDir.path() + "/default.cub", label, isd, "rw");

    projTestCube = new Cube();
    projTestCube->fromIsd(tempDir.path() + "/default.level2.cub", projLabel, isd, "rw");
  }


  void LineScannerCube::TearDown() {
    if (testCube->isOpen()) {
      testCube->close();
    }

    if (projTestCube->isOpen()) {
      projTestCube->close();
    }

    delete testCube;
    delete projTestCube;
  }


  void ThreeImageNetwork::SetUp() {
    TempTestingFiles::SetUp();

    FileName labelPath1("data/threeImageNetwork/cube1.pvl");
    FileName labelPath2("data/threeImageNetwork/cube2.pvl");
    FileName labelPath3("data/threeImageNetwork/cube3.pvl");

    isdPath1 = new FileName("data/threeImageNetwork/cube1.isd");
    isdPath2 = new FileName("data/threeImageNetwork/cube2.isd");
    isdPath3 = new FileName("data/threeImageNetwork/cube3.isd");

    threeImageOverlapFile = new FileName("data/threeImageNetwork/threeImageOverlaps.lis");
    twoImageOverlapFile = new FileName("data/threeImageNetwork/twoImageOverlaps.lis");

    cube1 = new Cube();
    cube1->fromIsd(tempDir.path() + "/cube1.cub", labelPath1, *isdPath1, "rw");

    ImagePolygon poly;
    coords = {{30, 0},
              {30, 10},
              {35, 10},
              {35, 0},
              {30, 0}};
    poly.Create(coords);
    cube1->write(poly);
    cube1->reopen("rw");

    cube2 = new Cube();
    cube2->fromIsd(tempDir.path() + "/cube2.cub", labelPath2, *isdPath2, "rw");

    coords = {{31, 1},
              {31, 11},
              {36, 11},
              {36, 1},
              {31, 1}};
    poly.Create(coords);
    cube2->write(poly);
    cube2->reopen("rw");

    cube3 = new Cube();
    cube3->fromIsd(tempDir.path() + "/cube3.cub", labelPath3, *isdPath3, "rw");

    cubeList = new FileList();
    cubeList->append(cube1->fileName());
    cubeList->append(cube2->fileName());
    cubeList->append(cube3->fileName());

    cubeListFile = tempDir.path() + "/cubes.lis";
    cubeList->write(cubeListFile);

    network = new ControlNet();
    network->ReadControl("data/threeImageNetwork/controlnetwork.net");
  }


  void ThreeImageNetwork::TearDown() {
    delete cubeList;
    delete network;

    delete cube1;
    delete cube2;
    delete cube3;

    delete isdPath1;
    delete isdPath2;
    delete isdPath3;

    delete threeImageOverlapFile;
    delete twoImageOverlapFile;
  }

  void ApolloNetwork::SetUp() {
    TempTestingFiles::SetUp();
    
    isdFile1 = new FileName("data/apolloNetwork/apolloImage1.isd");
    isdFile2 = new FileName("data/apolloNetwork/apolloImage2.isd");
    isdFile3 = new FileName("data/apolloNetwork/apolloImage3.isd");
    isdFile4 = new FileName("data/apolloNetwork/apolloImage4.isd");
    isdFile5 = new FileName("data/apolloNetwork/apolloImage5.isd");
    isdFile6 = new FileName("data/apolloNetwork/apolloImage6.isd");
    isdFile7 = new FileName("data/apolloNetwork/apolloImage7.isd");

    label1 = new FileName("data/apolloNetwork/apolloImage1.pvl");
    label2 = new FileName("data/apolloNetwork/apolloImage2.pvl");
    label3 = new FileName("data/apolloNetwork/apolloImage3.pvl");
    label4 = new FileName("data/apolloNetwork/apolloImage4.pvl");
    label5 = new FileName("data/apolloNetwork/apolloImage5.pvl");
    label6 = new FileName("data/apolloNetwork/apolloImage6.pvl");
    label7 = new FileName("data/apolloNetwork/apolloImage7.pvl");

    cube1 = new Cube();
    cube1->fromIsd(tempDir.path() + "/cube1.cub", *label1, *isdFile1, "rw");

    cube2 = new Cube();
    cube2->fromIsd(tempDir.path() + "/cube2.cub", *label2, *isdFile2, "rw");

    cube3 = new Cube();
    cube3->fromIsd(tempDir.path() + "/cube3.cub", *label3, *isdFile3, "rw");

    cube4 = new Cube();
    cube4->fromIsd(tempDir.path() + "/cube4.cub", *label4, *isdFile4, "rw");

    cube5 = new Cube();
    cube5->fromIsd(tempDir.path() + "/cube5.cub", *label5, *isdFile5, "rw");

    cube6 = new Cube();
    cube6->fromIsd(tempDir.path() + "/cube6.cub", *label6, *isdFile6, "rw");

    cube7 = new Cube();
    cube7->fromIsd(tempDir.path() + "/cube7.cub", *label7, *isdFile7, "rw");

    cubeList = new FileList();
    cubeList->append(cube1->fileName());
    cubeList->append(cube2->fileName());
    cubeList->append(cube3->fileName());
    cubeList->append(cube4->fileName());
    cubeList->append(cube5->fileName());
    cubeList->append(cube6->fileName());
    cubeList->append(cube7->fileName());

    cubeListFile = tempDir.path() + "/cubes.lis";
    cubeList->write(cubeListFile);

    network = new ControlNet("data/apolloNetwork/apolloNet.pvl");
    controlNetPath = tempDir.path() + "/apolloNet.net";
    network->Write(controlNetPath);
  }

  void ApolloNetwork::TearDown() {
    if (cube1->isOpen()) {
      cube1->close();
    }

    if (cube2->isOpen()) {
      cube1->close();
    }

    if (cube3->isOpen()) {
      cube1->close();
    }

    if (cube4->isOpen()) {
      cube1->close();
    }

    if (cube5->isOpen()) {
      cube1->close();
    }

    if (cube6->isOpen()) {
      cube1->close();
    }

    if (cube7->isOpen()) {
      cube1->close();
    }

    if (isdFile1) {
      delete isdFile1; 
    }

    if (isdFile2) {
      delete isdFile2;
    }

    if (isdFile3) {
      delete isdFile3; 
    }

    if (isdFile4) {
      delete isdFile4; 
    }

    if (isdFile5) {
      delete isdFile5; 
    }

    if (isdFile6) {
      delete isdFile6; 
    }

    if (isdFile7) {
      delete isdFile7; 
    }

    if (cube1) {
      delete cube1; 
    }

    if (cube2) {
      delete cube2; 
    }

    if (cube3) {
      delete cube3; 
    }

    if (cube4) {
      delete cube4; 
    }

    if (cube5) {
      delete cube5; 
    }

    if (cube6) {
      delete cube6; 
    }

    if (cube7) {
      delete cube7; 
    }

    if (cubeList) {
      delete cubeList; 
    }
  }

  void ObservationPair::SetUp() {
      FileName labelPathL = FileName("data/observationPair/observationImageL.pvl");
      FileName labelPathR = FileName("data/observationPair/observationImageR.pvl");

      isdPathL = new FileName("data/observationPair/observationImageL.isd");
      isdPathR = new FileName("data/observationPair/observationImageR.isd");

      cubeL = new Cube();
      cubeR = new Cube();

      cubeLPath = tempDir.path() + "/observationPairL.cub";
      cubeRPath = tempDir.path() + "/observationPairR.cub";

      cubeL->fromIsd(cubeLPath, labelPathL, *isdPathL, "rw");    
      cubeR->fromIsd(cubeRPath, labelPathR, *isdPathR, "rw");    

      cubeList = new FileList();
      cubeList->append(cubeL->fileName());
      cubeList->append(cubeR->fileName());

      cubeListFile = tempDir.path() + "/cubes.lis";
      cubeList->write(cubeListFile);

      cnetPath = "data/observationPair/observationPair.net";
      network = new ControlNet();
      network->ReadControl(cnetPath);
  }


  void ObservationPair::TearDown() {
    delete cubeList;
    delete network;

    if (cubeL) {
      delete cubeL;
    }
    
    if (cubeR) {
      delete cubeR;
    }

    delete isdPathL;
    delete isdPathR;

  }


  void MroCube::setInstrument(QString ikid, QString instrumentId, QString spacecraftName) {
    PvlGroup &kernels = testCube->label()->findObject("IsisCube").findGroup("Kernels");
    kernels.findKeyword("NaifFrameCode").setValue(ikid);    
    
    PvlGroup &inst = testCube->label()->findObject("IsisCube").findGroup("Instrument");
    std::istringstream iss(R"(
      Group = Instrument
        SpacecraftName              = "MARS RECONNAISSANCE ORBITER"
        InstrumentId                = HIRISE
        TargetName                  = Mars
        StartTime                   = 2008-05-17T09:37:24.7300819
        StopTime                    = 2008-05-17T09:37:31.0666673
        ObservationStartCount       = 895484264:44383
        SpacecraftClockStartCount   = 895484264:57342
        SpacecraftClockStopCount    = 895484272:12777
        ReadoutStartCount           = 895484659:31935
        CalibrationStartTime        = 2006-11-08T04:49:13.952
        CalibrationStartCount       = 847428572:51413
        AnalogPowerStartTime        = 2006-11-08T04:48:34.478
        AnalogPowerStartCount       = 847428533:20297
        MissionPhaseName            = "PRIMARY SCIENCE PHASE"
        LineExposureDuration        = 95.0625 <MICROSECONDS>
        ScanExposureDuration        = 95.0625 <MICROSECONDS>
        DeltaLineTimerCount         = 337
        Summing                     = 1
        Tdi                         = 128
        FocusPositionCount          = 2020
        PoweredCpmmFlag             = (On, On, On, On, On, On, On, On, On, On, On,
                                      On, On, On)
        CpmmNumber                  = 8
        CcdId                       = RED5
        ChannelNumber               = 0
        LookupTableType             = Stored
        LookupTableNumber           = 19
        LookupTableMinimum          = -9998
        LookupTableMaximum          = -9998
        LookupTableMedian           = -9998
        LookupTableKValue           = -9998
        StimulationLampFlag         = (Off, Off, Off)
        HeaterControlFlag           = (On, On, On, On, On, On, On, On, On, On, On,
                                      On, On, On)
        OptBnchFlexureTemperature   = 19.5881 <C>
        OptBnchMirrorTemperature    = 19.6748 <C>
        OptBnchFoldFlatTemperature  = 19.9348 <C>
        OptBnchFpaTemperature       = 19.5015 <C>
        OptBnchFpeTemperature       = 19.2415 <C>
        OptBnchLivingRmTemperature  = 19.4148 <C>
        OptBnchBoxBeamTemperature   = 19.5881 <C>
        OptBnchCoverTemperature     = 19.6748 <C>
        FieldStopTemperature        = 17.9418 <C>
        FpaPositiveYTemperature     = 18.8082 <C>
        FpaNegativeYTemperature     = 18.6349 <C>
        FpeTemperature              = 18.0284 <C>
        PrimaryMirrorMntTemperature = 19.5015 <C>
        PrimaryMirrorTemperature    = 19.6748 <C>
        PrimaryMirrorBafTemperature = 2.39402 <C>
        MsTrussLeg0ATemperature     = 19.6748 <C>
        MsTrussLeg0BTemperature     = 19.8482 <C>
        MsTrussLeg120ATemperature   = 19.3281 <C>
        MsTrussLeg120BTemperature   = 20.1949 <C>
        MsTrussLeg240ATemperature   = 20.2816 <C>
        MsTrussLeg240BTemperature   = 20.7151 <C>
        BarrelBaffleTemperature     = -13.8299 <C>
        SunShadeTemperature         = -33.9377 <C>
        SpiderLeg30Temperature      = 17.5087 <C>
        SpiderLeg120Temperature     = -9999
        SpiderLeg240Temperature     = -9999
        SecMirrorMtrRngTemperature  = 20.6284 <C>
        SecMirrorTemperature        = 20.455 <C>
        SecMirrorBaffleTemperature  = -11.1761 <C>
        IeaTemperature              = 25.4878 <C>
        FocusMotorTemperature       = 21.4088 <C>
        IePwsBoardTemperature       = 16.3696 <C>
        CpmmPwsBoardTemperature     = 17.6224 <C>
        MechTlmBoardTemperature     = 34.7792 <C>
        InstContBoardTemperature    = 34.4121 <C>
        DllLockedFlag               = (YES, YES)
        DllResetCount               = 0
        DllLockedOnceFlag           = (YES, YES)
        DllFrequenceCorrectCount    = 4
        ADCTimingSetting            = -9999
        Unlutted                    = TRUE
      End_Group
    )");
    
    PvlGroup newInstGroup; 
    iss >> newInstGroup; 
    
    newInstGroup.findKeyword("InstrumentId").setValue(instrumentId);
    newInstGroup.findKeyword("SpacecraftName").setValue(spacecraftName);

    inst = newInstGroup; 
    PvlObject &naifKeywords = testCube->label()->findObject("NaifKeywords");
    
    PvlKeyword startcc("SpacecraftClockStartCount", "33322515");
    PvlKeyword stopcc("SpaceCraftClockStopCount", "33322516");
    inst += startcc;
    inst += stopcc;  
    
    json nk; 
    nk["INS"+ikid.toStdString()+"_FOCAL_LENGTH"] = 11994.9988;
    nk["INS"+ikid.toStdString()+"_PIXEL_PITCH"] = 0.012;
    nk["INS"+ikid.toStdString()+"_TRANSX"] = {-89.496, -1.0e-06, 0.012};
    nk["INS"+ikid.toStdString()+"_TRANSY"] = {-12.001, -0.012, -1.0e-06};
    nk["INS"+ikid.toStdString()+"_ITRANSS"] = {-1000.86, -0.0087, -83.333};
    nk["INS"+ikid.toStdString()+"_ITRANSL"] = {7457.9, 83.3333, -0.0087};
    nk["INS"+ikid.toStdString()+"_OD_K"] = {-0.0048509, 2.41312e-07, -1.62369e-13};
    nk["BODY499_RADII"] = {3396.19, 3396.19, 3376.2};
    nk["CLOCK_ET_-74999_847428572:52459_COMPUTED"] = "8ed6ae8930f3bd41";
    nk["BODY_CODE"] = 499;
    nk["BODY_FRAME_CODE"] = 10014; 
    PvlObject newNaifKeywords("NaifKeywords", nk);
    naifKeywords = newNaifKeywords; 

    QString fileName = testCube->fileName();
   
    LineManager line(*testCube);
    for(line.begin(); !line.end(); line++) {
        for(int i = 0; i < line.size(); i++) {
          line[i] = (double)(i+1);
        }
        testCube->write(line);
    }
    testCube->reopen("rw");
  
    // need to remove old camera pointer 
    delete testCube;
    // This is now a MRO cube

    testCube = new Cube(fileName, "rw");

    // create a jitter file 
    QString jitter = R"(# Sample                 Line                   ET
-0.18     -0.07     264289109.96933
-0.11     -0.04     264289109.97
-0.05     -0.02     264289109.98
1.5     0.6     264289110.06
    )"; 
    
    jitterPath = tempDir.path() + "/jitter.txt"; 
    QFile jitterFile(jitterPath); 
    
    if (jitterFile.open(QIODevice::WriteOnly)) {
      QTextStream out(&jitterFile); 
      out << jitter;
      jitterFile.close(); 
    }
    else { 
      FAIL() << "Failed to create Jitter file" << std::endl;
    }
  }
}
