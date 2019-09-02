#include "Singleton.h"

std::shared_ptr<Singleton> Singleton::gInst = nullptr;
std::mutex Singleton::gMutex;

std::shared_ptr<Singleton> Singleton::GetInstance() {
    if (gInst == nullptr) {
        std::lock_guard<std::mutex> _lock_(gMutex);
        if (gInst == nullptr) { gInst.reset(new Singleton); }
    }
    return gInst;
}

void Singleton::setValue(int v) {
    std::lock_guard<std::mutex> _lock_(gMutex);
    mValue = v;
}

int Singleton::getValue() {
    std::lock_guard<std::mutex> _lock_(gMutex);
    return mValue;
}