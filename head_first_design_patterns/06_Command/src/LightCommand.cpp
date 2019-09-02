#include "LightCommand.h"

#include <iostream>

void LightCommand::execute() {
    if (bLightOn) {
        std::cout << "Light: TURN OFF" << std::endl;
    } else {
        std::cout << "Light: TURN ON" << std::endl;
    }
    bLightOn = !bLightOn;
}