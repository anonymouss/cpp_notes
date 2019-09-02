#ifndef __I_DOUGH_H__
#define __I_DOUGH_H__

#include <string>

class IDough {
public:
    virtual ~IDough() = default;
    virtual std::string getName() const { return mName; }

protected:
    std::string mName;
};

#endif  // __I_DOUGH_H__