#include "FanCommand.h"

#include <iostream>

void FanCommand::execute() {
    if (bFanOn) {
        std::cout << "Fan: TURN OFF" << std::endl;
    } else {
        std::cout << "Fan: TURN ON" << std::endl;
    }
    bFanOn = !bFanOn;
}