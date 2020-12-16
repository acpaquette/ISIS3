#include <QTemporaryDir>
#include <memory>

#include "thm2isis.h"
#include "Fixtures.h"
#include "Pvl.h"
#include "PvlGroup.h"
#include "TestUtilities.h"
#include "Histogram.h"
#include "Endian.h"
#include "PixelType.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Fixtures.h"

using namespace Isis;
using namespace testing;

static QString APP_XML = FileName("$ISISROOT/bin/xml/thm2isis.xml").expanded();

TEST_F(TempTestingFiles, FunctionalTestsThm2isisIr) {
  QString cubeFileName = tempDir.path() + "/thm2isisTEMP.cub";
  QVector<QString> args = {"from=data/themis/I00831002RDR_cropped.QUB",
                           "to=" + cubeFileName};

  UserInterface options(APP_XML, args);
  try {
   thm2isis(options);
  }
  catch (IException &e) {
    FAIL() << "Unable to ingest themis image: " << e.toString().toStdString().c_str() << std::endl;
  }
  std::unique_ptr<Cube> cube (new Cube(cubeFileName));
  Pvl *isisLabel = cube->label();

  // Dimensions Group
  EXPECT_EQ(cube->sampleCount(), 320);
  EXPECT_EQ(cube->lineCount(), 10);
  EXPECT_EQ(cube->bandCount(), 10);

  // Pixels Group
  EXPECT_EQ(cube->pixelType(), PixelType::Real);
  EXPECT_EQ(cube->byteOrder(), ByteOrder::Lsb);
  EXPECT_EQ(cube->base(), 0.0);
  EXPECT_EQ(cube->multiplier(), 1.0);

  // Instrument Group
  PvlGroup &inst = isisLabel->findGroup("Instrument", Pvl::Traverse);
  EXPECT_EQ(inst["SpacecraftName"][0].toStdString(), "MARS_ODYSSEY");
  EXPECT_EQ(inst["InstrumentId"][0].toStdString(), "THEMIS_IR");
  EXPECT_EQ(inst["TargetName"][0].toStdString(), "MARS");
  EXPECT_EQ(inst["StartTime"][0].toStdString(), "2002-02-20T22:57:57.253000");
  EXPECT_EQ(inst["StopTime"][0].toStdString(), "2002-02-20T23:00:56.983000");
  EXPECT_EQ(inst["SpacecraftClockCount"][0].toStdString(), "698713127.000");
  EXPECT_DOUBLE_EQ(double(inst["GainNumber"]), 16);
  EXPECT_DOUBLE_EQ(double(inst["OffsetNumber"]), 0);
  EXPECT_DOUBLE_EQ(double(inst["MissingScanLines"]), 0);
  EXPECT_DOUBLE_EQ(double(inst["GainNumber"]), 16);
  EXPECT_EQ(inst["TimeDelayIntegration"][0].toStdString(), "ENABLED");
  EXPECT_EQ(inst["SpacecraftClockOffset"][0].toStdString(), "0.0");
  EXPECT_EQ(inst["SpacecraftClockOffset"].unit(), "seconds");

  // BandBin Group
  PvlGroup &bandbin = isisLabel->findGroup("BandBin", Pvl::Traverse);
  std::vector<int> expectedOriginalBand = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> expectedCenter = {6.78, 6.78, 7.93, 8.56, 9.35, 10.21,
                                        11.04, 11.79, 12.57, 14.88};
  std::vector<double> expectedWidth = {1.01, 1.01, 1.09, 1.16, 1.2, 1.1,
                                       1.19, 1.07, 0.81, 0.87};
  std::vector<int> expectedFilterNum = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> expectedBandNum = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (int i = 0; i < expectedOriginalBand.size(); i++) {
    EXPECT_EQ(expectedOriginalBand[i], bandbin["OriginalBand"][i].toInt());
    EXPECT_EQ(expectedCenter[i], bandbin["Center"][i].toDouble());
    EXPECT_EQ(expectedWidth[i], bandbin["Width"][i].toDouble());
    EXPECT_EQ(expectedFilterNum[i], bandbin["FilterNumber"][i].toInt());
    EXPECT_EQ(expectedBandNum[i], bandbin["BandNumber"][i].toInt());
  }

  // Kernels Group
  PvlGroup &kernel = isisLabel->findGroup("Kernels", Pvl::Traverse);
  EXPECT_EQ(int(kernel["NaifFrameCode"]), -53031);


  Histogram* hist (cube->histogram(0));
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00050067109171232009);
  EXPECT_DOUBLE_EQ(hist->Sum(), 16.021474934794242);
  EXPECT_EQ(hist->ValidPixels(), 32000);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.00012358176146689913);
  delete hist;

  hist = cube->histogram(1);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00044544746567225956);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.4254318901512306);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.1793759250527171e-05);
  delete hist;

  hist = cube->histogram(2);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00044025198634699335);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.4088063563103788);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.2115190306691905e-05);
  delete hist;

  hist = cube->histogram(3);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00050481051051974645);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.6153936336631887);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.7345479518019523e-05);
  delete hist;

  hist = cube->histogram(4);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00050981284157387559);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.631401093036402);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.6970020502410073e-05);
  delete hist;

  hist = cube->histogram(5);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00053160175853918191);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.7011256273253821);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.6931465772319629e-05);
  delete hist;

  hist = cube->histogram(6);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00057632195762380434);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.8442302643961739);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.7766022638966021e-05);
  delete hist;

  hist = cube->histogram(7);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00059891219130804531);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.9165190121857449);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.9141699404527645e-05);
  delete hist;

  hist = cube->histogram(8);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00060846633206892878);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.947092262620572);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.9540049483839706e-05);
  delete hist;

  hist = cube->histogram(9);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00060713443390341131);
  EXPECT_DOUBLE_EQ(hist->Sum(), 1.942830188490916);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.5376997469565179e-05);
  delete hist;

  hist = cube->histogram(10);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.00018395143956695392);
  EXPECT_DOUBLE_EQ(hist->Sum(), 0.58864460661425255);
  EXPECT_EQ(hist->ValidPixels(), 3200);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 4.7876398382339314e-06);
  delete hist;
}

TEST_F(TempTestingFiles, FunctionalTestsThm2isisVis) {
  QString outFileName = "/thm2isisTEMP";
  QString cubeFileName = tempDir.path() + outFileName + ".cub";
  QVector<QString> args = {"from=data/themis/V00821003RDR_cropped.QUB",
                           "to=" + cubeFileName};

  UserInterface options(APP_XML, args);
  try {
   thm2isis(options);
  }
  catch (IException &e) {
    FAIL() << "Unable to ingest themis image: " << e.toString().toStdString().c_str() << std::endl;
  }
  std::unique_ptr<Cube> oddCube (new Cube(tempDir.path() + outFileName + ".odd.cub"));
  std::unique_ptr<Cube> evenCube (new Cube(tempDir.path() + outFileName + ".even.cub"));
  Pvl *oddLabel = oddCube->label();
  Pvl *evenLabel = evenCube->label();

  // Dimensions Group
  EXPECT_EQ(oddCube->sampleCount(), 1024);
  EXPECT_EQ(oddCube->lineCount(), 200);
  EXPECT_EQ(oddCube->bandCount(), 5);

  EXPECT_EQ(evenCube->sampleCount(), 1024);
  EXPECT_EQ(evenCube->lineCount(), 200);
  EXPECT_EQ(evenCube->bandCount(), 5);

  // Pixels Group
  EXPECT_EQ(oddCube->pixelType(), PixelType::Real);
  EXPECT_EQ(oddCube->byteOrder(), ByteOrder::Lsb);
  EXPECT_EQ(oddCube->base(), 0.0);
  EXPECT_EQ(oddCube->multiplier(), 1.0);

  EXPECT_EQ(evenCube->pixelType(), PixelType::Real);
  EXPECT_EQ(evenCube->byteOrder(), ByteOrder::Lsb);
  EXPECT_EQ(evenCube->base(), 0.0);
  EXPECT_EQ(evenCube->multiplier(), 1.0);

  // Instrument Group
  PvlGroup &inst = oddLabel->findGroup("Instrument", Pvl::Traverse);
  EXPECT_EQ(inst["SpacecraftName"][0].toStdString(), "MARS_ODYSSEY");
  EXPECT_EQ(inst["InstrumentId"][0].toStdString(), "THEMIS_VIS");
  EXPECT_EQ(inst["TargetName"][0].toStdString(), "MARS");
  EXPECT_EQ(inst["StartTime"][0].toStdString(), "2002-02-20T03:14:02.471000");
  EXPECT_EQ(inst["StopTime"][0].toStdString(), "2002-02-20T03:14:09.471000");
  EXPECT_EQ(inst["SpacecraftClockCount"][0].toStdString(), "698642092.025");
  EXPECT_DOUBLE_EQ(double(inst["ExposureDuration"]), 6.0);
  EXPECT_DOUBLE_EQ(double(inst["InterframeDelay"]), 1.0);
  EXPECT_DOUBLE_EQ(double(inst["SpatialSumming"]), 1);
  EXPECT_EQ(inst["SpacecraftClockOffset"][0].toStdString(), "0.0");
  EXPECT_EQ(inst["SpacecraftClockOffset"].unit(), "seconds");
  EXPECT_DOUBLE_EQ(double(inst["NumFramelets"]), 1);
  EXPECT_EQ(inst["Framelets"][0].toStdString(), "Odd");

  inst = evenLabel->findGroup("Instrument", Pvl::Traverse);
  EXPECT_EQ(inst["SpacecraftName"][0].toStdString(), "MARS_ODYSSEY");
  EXPECT_EQ(inst["InstrumentId"][0].toStdString(), "THEMIS_VIS");
  EXPECT_EQ(inst["TargetName"][0].toStdString(), "MARS");
  EXPECT_EQ(inst["StartTime"][0].toStdString(), "2002-02-20T03:14:02.471000");
  EXPECT_EQ(inst["StopTime"][0].toStdString(), "2002-02-20T03:14:09.471000");
  EXPECT_EQ(inst["SpacecraftClockCount"][0].toStdString(), "698642092.025");
  EXPECT_DOUBLE_EQ(double(inst["ExposureDuration"]), 6.0);
  EXPECT_DOUBLE_EQ(double(inst["InterframeDelay"]), 1.0);
  EXPECT_DOUBLE_EQ(double(inst["SpatialSumming"]), 1);
  EXPECT_EQ(inst["SpacecraftClockOffset"][0].toStdString(), "0.0");
  EXPECT_EQ(inst["SpacecraftClockOffset"].unit(), "seconds");
  EXPECT_DOUBLE_EQ(double(inst["NumFramelets"]), 1);
  EXPECT_EQ(inst["Framelets"][0].toStdString(), "Even");


  // BandBin Group
  PvlGroup &bandbin = oddLabel->findGroup("BandBin", Pvl::Traverse);
  std::vector<int> expectedOriginalBand = {1, 2, 3, 4, 5};
  std::vector<double> expectedCenter = {0.42, 0.54, 0.65, 0.75, 0.86};
  std::vector<double> expectedWidth = {0.05, 0.05, 0.05, 0.05, 0.04};
  std::vector<int> expectedFilterNum = {2, 5, 3, 4, 1};
  std::vector<int> expectedBandNum = {1, 2, 3, 4, 5};
  for (int i = 0; i < expectedOriginalBand.size(); i++) {
    EXPECT_EQ(expectedOriginalBand[i], bandbin["OriginalBand"][i].toInt());
    EXPECT_EQ(expectedCenter[i], bandbin["Center"][i].toDouble());
    EXPECT_EQ(expectedWidth[i], bandbin["Width"][i].toDouble());
    EXPECT_EQ(expectedFilterNum[i], bandbin["FilterNumber"][i].toInt());
    EXPECT_EQ(expectedBandNum[i], bandbin["BandNumber"][i].toInt());
  }

  bandbin = evenLabel->findGroup("BandBin", Pvl::Traverse);
  for (int i = 0; i < expectedOriginalBand.size(); i++) {
    EXPECT_EQ(expectedOriginalBand[i], bandbin["OriginalBand"][i].toInt());
    EXPECT_EQ(expectedCenter[i], bandbin["Center"][i].toDouble());
    EXPECT_EQ(expectedWidth[i], bandbin["Width"][i].toDouble());
    EXPECT_EQ(expectedFilterNum[i], bandbin["FilterNumber"][i].toInt());
    EXPECT_EQ(expectedBandNum[i], bandbin["BandNumber"][i].toInt());
  }

  // Kernels Group
  PvlGroup &kernel = oddLabel->findGroup("Kernels", Pvl::Traverse);
  EXPECT_EQ(int(kernel["NaifFrameCode"]), -53032);

  kernel = evenLabel->findGroup("Kernels", Pvl::Traverse);
  EXPECT_EQ(int(kernel["NaifFrameCode"]), -53032);

  Histogram* hist (oddCube->histogram(0));
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0032050829203333175);
  EXPECT_DOUBLE_EQ(hist->Sum(), 2407.2288086430635);
  EXPECT_EQ(hist->ValidPixels(), 751066);
  EXPECT_EQ(hist->NullPixels(), 272934);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.0012757993805892806);
  delete hist;

  hist = oddCube->histogram(1);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0012146869906190147);
  EXPECT_DOUBLE_EQ(hist->Sum(), 228.48262293543667);
  EXPECT_EQ(hist->ValidPixels(), 188100);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 2.5492910673775158e-05);
  delete hist;

  hist = oddCube->histogram(2);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0030423641729513695);
  EXPECT_DOUBLE_EQ(hist->Sum(), 572.26870093215257);
  EXPECT_EQ(hist->ValidPixels(), 188100);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 8.1132094641407011e-05);
  delete hist;

  hist = oddCube->histogram(3);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0045459914483793511);
  EXPECT_DOUBLE_EQ(hist->Sum(), 849.0366388480179);
  EXPECT_EQ(hist->ValidPixels(), 186766);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.00014736498719936585);
  delete hist;

  hist = oddCube->histogram(4);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0040267987555951965);
  EXPECT_DOUBLE_EQ(hist->Sum(), 757.44084592745639);
  EXPECT_EQ(hist->ValidPixels(), 188100);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.00014715790870523248);
  delete hist;

  hist = oddCube->histogram(5);
  EXPECT_DOUBLE_EQ(hist->Average(), -1.7976931348623149e+308);
  EXPECT_DOUBLE_EQ(hist->Sum(), 0);
  EXPECT_EQ(hist->ValidPixels(), 0);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), -1.7976931348623149e+308);
  delete hist;

  hist = evenCube->histogram(0);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0031812268601306222);
  EXPECT_DOUBLE_EQ(hist->Sum(), 125.25762639078312);
  EXPECT_EQ(hist->ValidPixels(), 39374);
  EXPECT_EQ(hist->NullPixels(), 984626);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.0011403444106711229);
  delete hist;

  hist = evenCube->histogram(1);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0012088832911103964);
  EXPECT_DOUBLE_EQ(hist->Sum(), 9.5743556655943394);
  EXPECT_EQ(hist->ValidPixels(), 7920);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 1.9940037117194649e-05);
  delete hist;

  hist = evenCube->histogram(2);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0030467073489838948);
  EXPECT_DOUBLE_EQ(hist->Sum(), 24.129922203952447);
  EXPECT_EQ(hist->ValidPixels(), 7920);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 6.7849469699949677e-05);
  delete hist;

  hist = evenCube->histogram(3);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0045480986605348729);
  EXPECT_DOUBLE_EQ(hist->Sum(), 34.993071094155312);
  EXPECT_EQ(hist->ValidPixels(), 7694);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.00011021847268104142);
  delete hist;

  hist = evenCube->histogram(4);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0040220504221235957);
  EXPECT_DOUBLE_EQ(hist->Sum(), 31.854639343218878);
  EXPECT_EQ(hist->ValidPixels(), 7920);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 0.00011120636199309424);
  delete hist;

  hist = evenCube->histogram(5);
  EXPECT_DOUBLE_EQ(hist->Average(), 0.0031193987479623915);
  EXPECT_DOUBLE_EQ(hist->Sum(), 24.705638083862141);
  EXPECT_EQ(hist->ValidPixels(), 7920);
  EXPECT_DOUBLE_EQ(hist->StandardDeviation(), 3.7732129681642477e-05);
  delete hist;
}
