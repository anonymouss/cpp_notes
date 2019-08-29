#ifndef __I_DISPLAY_ELEMENT_H__
#define __I_DISPLAY_ELEMENT_H__

class IDisplayElement {
public:
    virtual ~IDisplayElement() = default;

    virtual void display() const = 0;
};

#endif  // __I_DISPLAY_ELEMENT_H__