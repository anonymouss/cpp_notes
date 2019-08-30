#ifndef __PEPPERON_PIZZA_H__
#define __PEPPERON_PIZZA_H__

#include "IPizza.h"

class PepperonPizza : public IPizza {
public:
    explicit PepperonPizza(const char *name);
    virtual ~PepperonPizza() = default;
};

#endif  // __PEPPERON_PIZZA_H__