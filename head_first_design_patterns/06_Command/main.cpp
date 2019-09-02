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

    // XXX: 此处应当是 uniqe_ptr 更合理，即独占所有权，插入 controller 后交出所有权（如果插入成功，否则保
    // 留所有权）。但是，unique_ptr 要么按值传参，要么按引用传参。按值需要显式 std::move，则所有权必定移入
    // 函数内，无法处理保留所有权的情况。引用传参则 std::unique_ptr<LightCommand> 与
    // std::unique_ptr<ICommand> 类型不匹配
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