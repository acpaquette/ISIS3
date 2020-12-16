#include "Isis.h"

#include "thm2isis.h"

#include "Application.h"

using namespace std;
using namespace Isis;

void IsisMain() {
  UserInterface &ui = Application::GetUserInterface();
  thm2isis(ui);
}
