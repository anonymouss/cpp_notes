#ifndef __REMOTE_CONTROLLER_H__
#define __REMOTE_CONTROLLER_H__

#include "ICommand.h"

#include <memory>
#include <vector>

class RemoteController {
public:
    RemoteController();
    virtual ~RemoteController() = default;
    bool plugin(int slot, std::shared_ptr<ICommand> cmd);
    bool plugoff(int slot, std::shared_ptr<ICommand> cmd);
    void pressButton(int slot);

private:
    std::vector<std::shared_ptr<ICommand>> mCommands;
};

#endif  // __REMOTE_CONTROLLER_H__