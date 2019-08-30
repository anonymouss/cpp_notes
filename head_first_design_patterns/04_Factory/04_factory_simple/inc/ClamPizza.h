#ifndef __CLAM_PIZZA_H__
#define __CLAM_PIZZA_H__

#include "IPizza.h"

class ClamPizza : public IPizza {
public:
    explicit ClamPizza(const char *name);
    virtual ~ClamPizza() = default;
};

#endif  // __CLAM_PIZZA_H__