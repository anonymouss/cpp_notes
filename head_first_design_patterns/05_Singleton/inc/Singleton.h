#ifndef __SINGLETON_H__
#define __SINGLETON_H__

#include <memory>
#include <mutex>

class Singleton {
public:
    virtual ~Singleton() = default;
    static std::shared_ptr<Singleton> GetInstance();
    void setValue(int v);
    int getValue();

private:
    Singleton() = default;
    static std::shared_ptr<Singleton> gInst;
    static std::mutex gMutex;

    int mValue;
};

#endif  // __SINGLETON_H__