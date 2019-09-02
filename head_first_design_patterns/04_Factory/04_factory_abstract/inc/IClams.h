#ifndef __I_CLAMS_H__
#define __I_CLAMS_H__

#include <string>

class IClams {
public:
    virtual ~IClams() = default;
    virtual std::string getName() const { return mName; }

protected:
    std::string mName;
};

#endif  // __I_CLAMS_H__