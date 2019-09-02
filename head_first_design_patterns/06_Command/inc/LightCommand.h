#ifndef __LIGHT_COMMAND_H__
#define __LIGHT_COMMAND_H__

#include "ICommand.h"

class LightCommand : public ICommand {
public:
    LightCommand() = default;
    virtual ~LightCommand() = default;
    virtual void execute() final;

private:
    bool bLightOn = false;
};

#endif  // __LIGHT_COMMAND_H__