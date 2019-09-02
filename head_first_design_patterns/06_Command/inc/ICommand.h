#ifndef __I_COMMAND_H__
#define __I_COMMAND_H__

class ICommand {
public:
    virtual ~ICommand() = default;
    virtual void execute() = 0;
};

#endif  // __I_COMMAND_H__