`std::thread` works with any callable type. so we can pass an instance of a class with a function call operator to it.

向线程函数传递参数：默认情况下，这些参数都会被**拷贝**到新线程的内部存储以便访问，然后以右值的方式传递给线程函数（可调用对象）。即使线程函数本身期待这些参数以引用传递也是这样。
```cpp
void f(int, const std::string &s);
std::thread t(f, 3, "hello");
// 虽然 f 接收 std::string 引用传参，这里字符串字面值 "hello" 还是会以 const char * 的形式传给新线程构
// 造，并且随后在新线程种被转换为 std::string。这一点在参数是自动变量的指针时尤为重要

// std::ref/cref
```

