#ifndef demprep_h // Change this to your app name in all lower case suffixed with _h (e.g. campt_h, cam2map_h etc.)
#define demprep_h

#include "Cube.h"
#include "UserInterface.h"

namespace Isis{
  extern void demprep(UserInterface &ui, Pvl *log);
}

#endif
