#ifndef __EFFECTIVE_MODERN_CPP_UTILS_H__
#define __EFFECTIVE_MODERN_CPP_UTILS_H__

#include <cstdio>
#include <string>

#ifdef HAS_BOOST
#include <boost/type_index.hpp>

#define ECHO_TYPE(T, U)                                                            \
    {                                                                              \
        using boost::typeindex::type_id_with_cvr;                                  \
        printf("%-5s = %-10s\n", #U, type_id_with_cvr<T>().pretty_name().c_str()); \
    }

#define ECHO_TYPE_T(T) ECHO_TYPE(T, T)
#define ECHO_TYPE_PARAM(param) ECHO_TYPE(decltype(param), param)

#define DESCRIBE()                                    \
    {                                                 \
        printf("%s\n", std::string(30, '-').c_str()); \
        ECHO_TYPE_T(T);                               \
        ECHO_TYPE_PARAM(param);                       \
        printf("%s\n", std::string(30, '-').c_str()); \
    }
#else
#define NO_BOOST_WARNING() \
    { printf("WARNING: No boost library supported\n"); }

#define ECHO_TYPE(T, U) NO_BOOST_WARNING()
#define ECHO_TYPE_T(T) NO_BOOST_WARNING()
#define ECHO_TYPE_PARAM(param) NO_BOOST_WARNING()
#define DESCRIBE() NO_BOOST_WARNING()
#endif  // HAS_BOOST

#define INFO(msg) \
    { printf("\n%s\n", msg); }

#endif  // __EFFECTIVE_MODERN_CPP_UTILS_H__
