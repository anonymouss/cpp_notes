#ifndef __MAX_1_H__
#define __MAX_1_H__

template <typename T>
T max(T a, T b) {
    return b < a ? a : b;
}

#endif  // __MAX_1_H__