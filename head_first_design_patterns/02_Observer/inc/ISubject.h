#ifndef __I_SUBJECT_H__
#define __I_SUBJECT_H__

class IObserver;

class ISubject {
public:
    virtual ~ISubject() = default;

    virtual void registerObserver(IObserver* observer) = 0;
    virtual void removeObserver(IObserver* observer) = 0;
    virtual void notifyObserverAll() = 0;
};

#endif  // __I_SUBJECT_H__