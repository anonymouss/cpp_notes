#ifndef __THIN_CRUST_DOUGH_H__
#define __THIN_CRUST_DOUGH_H__

#include "IDough.h"

class ThinCrustDough : public IDough {
public:
    ThinCrustDough() { mName = "<thin crust dough>"; }
};

#endif  // __THIN_CRUST_DOUGH_H__