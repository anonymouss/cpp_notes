#ifndef __I_DUCK_H__
#define __I_DUCK_H__

class IDuck {
public:
    virtual ~IDuck() = default;
    virtual void quack() = 0;
    virtual void fly() = 0;
};

#endif  // __I_DUCK_H__