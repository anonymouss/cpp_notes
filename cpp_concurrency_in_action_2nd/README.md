`std::thread` works with any callable type. so we can pass an instance of a class with a function call operator to it.

向线程函数传递参数：默认情况下，这些参数都会被**拷贝**到新线程的内部存储以便访问，然后以右值的方式传递给线程函数（可调用对象）。即使线程函数本身期待这些参数以引用传递也是这样。
```cpp
void f(int, const std::string &s);
std::thread t(f, 3, "hello");
// 虽然 f 接收 std::string 引用传参，这里字符串字面值 "hello" 还是会以 const char * 的形式传给新线程构
// 造，并且随后在新线程种被转换为 std::string。这一点在参数是自动变量的指针时尤为重要

// std::ref/cref
```

互斥锁可以保护对象，但是要注意，如果被保护对象以指针或引用方式从函数返回，那么它可能就不在互斥锁的保护范围内了。需要注意接口定义。

锁的粒度太细容易产生竞争条件，太大又会降低多线程带来的收益。在适当的粒度上锁！

避免死锁的一些方法：

- 避免嵌套锁。
- 持有锁时避免调用用户提供的代码。
- 总是以相同的顺序上锁。这并不容易（如swap），`std::scoped_lock`可以避免多个锁同时锁定时的死锁
- 使用层次锁

其他：
- 初始化期间保护数据：`std::call_once/std::once_flag` 保证只会调用一次
- 读写锁：`std::shared_mutex/std::shared_lock`。`lock_shared()`可以多个线程共享锁定，但是如果有线程进行了排他锁定`lock()`，该调用会阻塞。
- 递归锁/可重入锁：`std::recursive_mutex`，同一个线程可以在不释放的情况下多次上锁。（上锁几次也要释放几次）

`std::future/std::shared_future` 一次性等待。`std::future/std::async/std::packaged_task/std::promise`