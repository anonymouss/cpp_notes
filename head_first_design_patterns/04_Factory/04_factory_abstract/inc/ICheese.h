#ifndef __I_CHEESE_H__
#define __I_CHEESE_H__

#include <string>

class ICheese {
public:
    virtual ~ICheese() = default;
    virtual std::string getName() const { return mName; }

protected:
    std::string mName;
};

#endif  // __I_CHEESE_H__