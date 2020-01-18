#ifndef __TEST_TMPL_ARGS_H__
#define __TEST_TMPL_ARGS_H__

#include <list>

/*
template <typename T, template <typename> class Cont>
struct Relv1 {
    Cont<T> c;
};
*/

template <typename T, template <typename ...> class Cont>
struct Relv2 {
    Cont<T> c;
};

#endif // __TEST_TMPL_ARGS_H__