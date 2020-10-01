#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Cube.h"
#include "FileName.h"
#include "geos/operation/distance/DistanceOp.h"
#include "geos/util/IllegalArgumentException.h"
#include "geos/geom/Point.h"
#include "geos/opOverlay.h"
#include "IException.h"
#include "ImageOverlapThread.h"
#include "ImagePolygon.h"
#include "PolygonTools.h"
#include "Progress.h"
#include "SerialNumberList.h"

#include "QMessageBox"

using namespace std;

namespace Isis {
  /**
   * Create FindImageOverlaps object.
   *
   * Create an empty FindImageOverlaps object.
   *
   * @param continueOnError Whether or not this class throws exceptions in
   *                        addition to logging errors.
   * @see automaticRegistration.doc
   */
  ImageOverlapThread::ImageOverlapThread(ImageOverlap *polygon, QList<ImageOverlap *> *lonLatOverlaps, QMutex *lonLatOverlapsMutex) {
    p_threadPolygon = polygon;
    p_lonLatOverlaps = lonLatOverlaps;
    p_lonLatOverlapsMutex = lonLatOverlapsMutex;
  }

  void ImageOverlapThread::dummyRun() {
    p_lonLatOverlapsMutex->lock();
    std::cout << "SIZE ON INSERT: " << p_lonLatOverlaps->length() << '\n';
    p_lonLatOverlaps->push_back(p_threadPolygon);
    std::cout << "SIZE AFTER INSERT: " << p_lonLatOverlaps->length() << '\n';
    p_lonLatOverlapsMutex->unlock();
  }
}
