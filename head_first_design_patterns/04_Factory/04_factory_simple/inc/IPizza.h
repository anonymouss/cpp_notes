#ifndef __I_PIZZA_H__
#define __I_PIZZA_H__

#include <iostream>
#include <string>

class IPizza {
public:
    IPizza() = default;
    virtual ~IPizza() = default;

    virtual void prepare() { std::cout << ">> Preparing " << mName << std::endl; }
    virtual void bake() { std::cout << ">> Baking " << mName << std::endl; }
    virtual void cut() { std::cout << ">> Cutting " << mName << std::endl; }
    virtual void box() { std::cout << ">> Boxing " << mName << std::endl; }

protected:
    std::string mName;
};

#endif  // __I_PIZZA_H__