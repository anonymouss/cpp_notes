#ifndef __I_SAUCE_H__
#define __I_SAUCE_H__

#include <string>

class ISauce {
public:
    virtual ~ISauce() = default;
    virtual std::string getName() const { return mName; }

protected:
    std::string mName;
};

#endif  // __I_SAUCE_H__