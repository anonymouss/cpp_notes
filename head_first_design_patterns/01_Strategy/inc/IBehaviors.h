#ifndef __I_BEHAVIORS_H__
#define __I_BEHAVIORS_H__

// interface
class IFlyBehavior {
public:
    virtual void fly() = 0;
    virtual ~IFlyBehavior() = default;
};

class IQuackBehavior {
public:
    virtual void quack() = 0;
    virtual ~IQuackBehavior() = default;
};

#endif  // __I_BEHAVIORS_H__