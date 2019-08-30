#ifndef _I_BEVARAGE_H__
#define _I_BEVARAGE_H__

#include <string>

//
class IBeverage {
public:
    std::string mDescription = "UNKNOWN";

    virtual std::string getDescription() const { return mDescription; }
    virtual double calculateCost() = 0;

    virtual ~IBeverage() = default;
};

#endif  // _I_BEVARAGE_H__