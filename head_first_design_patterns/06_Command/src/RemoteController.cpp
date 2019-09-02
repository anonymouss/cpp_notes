#include "RemoteController.h"

#include <iostream>

constexpr int SLOT_NUMS = 3;

RemoteController::RemoteController() {
    mCommands = std::vector<std::shared_ptr<ICommand>>(SLOT_NUMS);
}

bool RemoteController::plugin(int slot, std::shared_ptr<ICommand> cmd) {
    if (slot < 0 || slot >= SLOT_NUMS) {
        std::cout << "ERROR: incorrect slot index : " << slot << std::endl;
        return false;
    }

    if (mCommands[slot]) {
        std::cout << "ERROR: slot [" << slot << "] is already connected" << std::endl;
        return false;
    }

    mCommands[slot] = cmd;
    return true;
}

bool RemoteController::plugoff(int slot, std::shared_ptr<ICommand> cmd) {
    if (slot < 0 || slot >= SLOT_NUMS) {
        std::cout << "ERROR: incorrect slot index : " << slot << std::endl;
        return false;
    }

    if (!mCommands[slot]) {
        std::cout << "ERROR: slot [" << slot << "] is already disconnected" << std::endl;
        return false;
    }

    cmd = mCommands[slot];
    mCommands[slot] = nullptr;
    return true;
}

void RemoteController::pressButton(int slot) {
    if (slot < 0 || slot >= SLOT_NUMS) {
        std::cout << "ERROR: incorrect slot index : " << slot << std::endl;
        return;
    }

    if (!mCommands[slot]) {
        std::cout << "ERROR: slot [" << slot << "] is empty" << std::endl;
        return;
    }

    mCommands[slot]->execute();
}