#ifndef __FAN_COMMAND_H__
#define __FAN_COMMAND_H__

#include "ICommand.h"

class FanCommand : public ICommand {
public:
    FanCommand() = default;
    virtual ~FanCommand() = default;
    virtual void execute() final;

private:
    bool bFanOn = false;
};

#endif  // __FAN_COMMAND_H__