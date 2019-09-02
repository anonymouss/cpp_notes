#ifndef __MARINARA_SAUCE_H__
#define __MARINARA_SAUCE_H__

#include "ISauce.h"

class MarinaraSauce : public ISauce {
public:
    MarinaraSauce() { mName = "<marinara sauce>"; }
};

#endif  // __MARINARA_SAUCE_H__