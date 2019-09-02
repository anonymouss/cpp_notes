#ifndef __FROZEN_CLAMS_H__
#define __FROZEN_CLAMS_H__

#include "IClams.h"

class FrozenClams : public IClams {
public:
    FrozenClams() { mName = "<frozen clams>"; }
};

#endif  // __FROZEN_CLAMS_H__