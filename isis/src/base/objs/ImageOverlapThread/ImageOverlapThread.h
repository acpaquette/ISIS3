#ifndef ImageOverlapThread_h
#define ImageOverlapThread_h
/**
 * @file
 * $Revision: 1.17 $
 * $Date: 2009/06/01 15:18:11 $
 *
 *   Unless noted otherwise, the portions of Isis written by the USGS are
 *   public domain. See individual third-party library and package descriptions
 *   for intellectual property information, user agreements, and related
 *   information.
 *
 *   Although Isis has been used by the USGS, no warranty, expressed or
 *   implied, is made by the USGS as to the accuracy and functioning of such
 *   software and related material nor shall the fact of distribution
 *   constitute any such warranty, and no responsibility is assumed by the
 *   USGS in connection therewith.
 *
 *   For additional information, launch
 *   $ISISROOT/doc/documents/Disclaimers/Disclaimers.html
 *   in a browser or see the Privacy &amp; Disclaimers page on the Isis website,
 *   http://isis.astrogeology.usgs.gov, and the USGS privacy and disclaimers on
 *   http://www.usgs.gov/privacy.html.
 */

#include <vector>
#include <string>

#include <QList>
#include <QThread>
#include <QMutex>

#include "geos/geom/MultiPolygon.h"
#include "geos/geom/LinearRing.h"
#include "geos/util/GEOSException.h"

#include "ImageOverlap.h"
#include "IException.h"
#include "PvlGroup.h"

namespace Isis {

  // Forward declarations
  class SerialNumberList;

  /**
    */
  class ImageOverlapThread : public QThread {
    public:
      ImageOverlapThread(ImageOverlap *polygon, QList<ImageOverlap *> *lonLatOverlaps, QMutex *lonLatOverlapsMutex);

    private:
      //! Find overlaps is all the threaded calculate does
      void run() {
        dummyRun();
      }
      void dummyRun();

    ImageOverlap *p_threadPolygon;
    QList<ImageOverlap *> *p_lonLatOverlaps;

    QMutex *p_lonLatOverlapsMutex;
  };
};

#endif
