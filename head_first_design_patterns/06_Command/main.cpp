#include "FanCommand.h"
#include "LightCommand.h"
#include "RemoteController.h"

#include <iostream>
#include <memory>

int main() {
    RemoteController controller;
    std::cout << "\nplug in light and fan" << std::endl;
    auto light = std::make_shared<LightCommand>();
    auto fan = std::make_shared<FanCommand>();

    auto reset_if_plugged = [](auto &ptr, bool ret) {
        if (ret) {
            ptr = nullptr;
        } else {
            std::cout << "FAILED PLUGIN" << std::endl;
        }
    };

    auto ret1 = controller.plugin(1, light);
    reset_if_plugged(light, ret1);
    auto ret2 = controller.plugin(2, fan);
    reset_if_plugged(fan, ret2);

    for (int i = 0; i < 4; ++i) { controller.pressButton(i); }

    std::cout << "\nremove light and fan" << std::endl;
    controller.plugoff(1, light);
    controller.plugoff(2, fan);

    for (int i = 0; i < 4; ++i) { controller.pressButton(i); }
}

/**
 * output
 *
 * plug in light and fan
 * ERROR: slot [0] is empty
 * Light: TURN ON
 * Fan: TURN ON
 * ERROR: incorrect slot index : 3
 *
 * remove light and fan
 * ERROR: slot [0] is empty
 * ERROR: slot [1] is empty
 * ERROR: slot [2] is empty
 * ERROR: incorrect slot index : 3
 */