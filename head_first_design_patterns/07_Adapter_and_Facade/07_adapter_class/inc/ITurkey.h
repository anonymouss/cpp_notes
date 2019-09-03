#ifndef __I_TURKEY_H__
#define __I_TURKEY_H__

class ITurkey {
public:
    virtual ~ITurkey() = default;
    virtual void gobble() = 0;
    virtual void fly() = 0;
};

#endif  // __I_TURKEY_H__