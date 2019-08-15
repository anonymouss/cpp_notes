# Learning Notes of < Effective Modern C++ >

> **Required**: [`boost`](https://www.boost.org/) C++ library, and [compiler supports at least `C++14`](https://zh.cppreference.com/w/cpp/compiler_support).

## 01. [Deducing Types](./source/01_deducing_types.cpp)

### **模板类型推断的基本原则**

```c++
// 伪代码
templte <typename T>
void f(ParamType param);

f(expr)
```

1. 形参类型为普通引用（非通用引用, aka. universal reference/forwarding reference）。

    - 忽略 `expr` 中的引用部分，与 `ParamType param` 匹配，`T` 为剩下的部分

2. 形参类型为通用引用。

    - 若 `expr` 为左值，`T` 与 `ParamType` 都会被推断为左值引用

    - 若 `expr` 为右值，则与 `1.` 同

3. 形参类型既不是指针也不是引用

    - 若 `expr` 是引用，忽略引用部分

    - 忽略 `cv` 限定符

4. **数组形参**

    - 注意退化为指针（`T arr[]` 与 `T *arr` 都会退化）

    - 形参为引用，不会退化（还可以获得实参数组维度）

5. **函数形参**

    - 与数组形参类似

### **`auto` 类型推断**

`auto` 基本相当于模板类型推断中的 `T`。但是要注意从 *初始化列表* 推断的情形。

**NOTE:**

- 模板实参传递 `std::initializer_list<T>` 形参必须显式声明。这点与 `auto` 不同。

- `C++14` 起，`auto` 可以推断函数返回值类型，但是使用的是模板推断规则，所以不能推断 `std::initializer_list<T>`

- `auto` in lambda, ditto

### **`decltype` 类型推断**

- `decltype` 可以获得精确的类型（保留`cv`限定符）

- `decltype(e)` 推断规则

    1. `e` 没有 `()` 括起来，推断结果即其声明类型

    2. `e` 是 `T` 类型的 `xvalue`，推断结果为 `T &&`

    3. `e` 是 `T` 类型的 `lvalue`，推断结果为 `T &`。（同时满足 `1.`，使用 `1.`）

    4. `e` 是 `T` 类型的 `prvalue`，推断结果为 `T`。*（`decltype((a + b))` 为 `T` 不是 `T &`，因为 `a + b` 为纯右值）*


## 02. [`auto`](./source/02_auto.cpp)

### 优先使用 `auto` 而非显式的类型声明

- 可以在编译期避免变量未初始化，否则 `auto` 无法推断类型

- 可以简化冗余的声明（显而易见）

- 可以获得只有编译器知道的类型（闭包）

- avoid `type shortcuts` （类型截断？）--> 不知道确切的类型而想当然的指定。

    1. 典型的如：`int size = vec.size();`，实际为`std::vector<T>::size_type`（通常为 `std::size_t`）。

### 当 `auto` 推断出的类型不是期望类型时，应当使用显式类型初始化的方法

- 翻译的真拗口。其实就是，因为有一些类存在代理类型（如 `vector<bool>`），这会让 `auto` 在表达式初始化时推导出的类型不正确（得到的是代理类型）。这时候应当使用显式类型初始化，使 `auto` 可以正确推导。

## 03. [Moving to Modern C++](./source/03_moving_to_modern_cpp.cpp)

### 注意区分创建对象时的 `()` 和 `{}`

- `{}` 更加通用，并且可以防止隐式的类型截断（implicit narrowing conversion）

- `()` 初始化有可能和函数声明混淆，如 `A a()`，函数声明还是显式调用无参构造函数？`A a{}` 就没有歧义。

- ***drawbacks***: 当定义了接收 `std::initializer_list<T>` 参数的构造函数时，`{}` 初始化会优先匹配该构造函数！即使 `T` 不相同，也会隐式转换。（注意，拷贝或移动构造的情况也有可能）。一个很重要的例子就是 `std::vector<IntType>` 通过两个参数构造时，可变参数模板内使用时要格外小心。

### 使用 `nullptr`，废弃 `0` 和 `NULL`

- 这是一个毫无悬念的选择。`0` 和 `NULL` 本质上是整型类型，某些情况下会有歧义影响重载决议。并且模板推导只会从它们推导出整型类型，而非指针类型。

### 优先使用类型别名，尽量避免`typedef`

- 直观的，类型别名定义比使用 `typedef` 更加清晰

```C++
// function ptr
typedef void(*FP)(int, const std::string &);
using FP = void(*)(int, const std::string &);
```
- 类型别名可以模板化（别名模板 alias templates），`typedef` 不可以

```C++
template <typename T>
using MyAllocList = std::list<T, MyAlloc<T>>;

MyAllocList<MyType> lst;

template <typename T>
struct MyAllocList {
    typedef std::list<T, MyAlloc<T>> type;
};

MyAllocList<MyType>::type lst
```

- 用在模板内部时，`typedef` 因为有 `::type` 是个依赖类型（待决名，因为模板不知道它代表的是类型）需要加 `typename` 前缀

```C++
template <typename T>
class Widget {
private:
    typename MyAllocList<T>::type list; // when using `typedef`
    MyAllocList<T> list;                // when using `using`
};

// this can be seen in type_traits
// eg.
// std::remove_const<T>::type;  // since C++11
// std::remove_const_t<T>;      // since C++14
```

### 尽量使用作用域枚举类型

- 作用域枚举类型的枚举项只对枚举内可见，只能通过 `cast` 转为其他类型

- 作用域和无作用域枚举类型都可以指定底层数据类型，作用域枚举类型默认是 `int`，无作用域枚举类型没有（由编译器决定） --> 因此作用域枚举类型可以前置声明，无作用域枚举类型只有指定底层类型的情况下可以前置声明。

### 使用 `delete` 来弃置函数，不要使用 `private`

- `delete` 适用于所有函数，不仅仅是成员函数（trick: 过滤模板）

- `member function ref-qualifier`

### 虚函数覆盖请在声明末尾加上 `override`

- `override` 可以告诉编译期这是一个派生类覆盖函数，请检查和基类函数签名是否一致（如果没有，可能出现函数隐藏的情况）

### 优先使用 `const_iterator` 而不是 `iterator`

- `cbegin()`，`cend()`

- 在通用代码中，使用非成员函数版本的 `begin, end, rbegin, etc.`

### 如果确定函数不会抛出异常，请在声明时加上 `noexcept`

- `noexcept` 是函数接口的一部分

### 在任何能用 `constepxr` 的地方使用 `constepxr`

- `constepxr` **变量** 必须在编译期求得

- `constepxr` **函数** 如果传入参数是编译期可求得，函数的返回结果也必须可以在编译期求得；如果传入参数是运行时才能知道，函数就和普通函数一样

    1. `C++11` 的 `constepxr` 函数只允许包含一条可执行语句（有且只有一条 `return` 语句），`C++14`放宽了这一限制

    2. `constepxr` 只能返回 `literal` 类型，基本类型满足这个要求，自定义类型要求构造函数也是 `constepxr` 的（成员也要有 `constepxr` 构造函数）

    3. `constexpr` 构造函数又要求其每个参数都是 `literal` 类型

    4. `constexpr` 是函数接口的一部分

- [more details](https://zh.cppreference.com/w/cpp/language/constexpr)

### 确保 `const` 成员函数线程安全

- 确保 `const` 成员函数是线程安全的，除非它不会在多线程中使用

- 可以使用 `std::mutex`，重量级；也可以使用 `std::atomic`，轻量级但如果函数中有多个变量需要原子操作，还是使用 `std::mutex` 来保证线程安全吧

### 理解特殊的成员函数

- 默认构造函数，默认析构函数，拷贝构造函数，拷贝赋值运算符，移动构造函数，移动赋值运算符

```C++
class Widget {
public:
    Widget();
    virtual ~Widget() noexcept;
    Widget(const Widget &w);
    Widget &operator=(const Widget &w);
    Widget(Widget &&w);
    Widget &operator=(Widget &&w);
};
```

- **Rule of Three for `C++98`**: 如果自定义了*拷贝构造*，*拷贝赋值*，*析构*中的一个，其余几个也得手动定义（删除也可以）。否则编译器会自动生成默认的（复杂的类，有可能出问题）

- 默认移动操作只有在需要时才生成，但还有以下前提：

    1. 没有定义拷贝操作

    2. 没有定义移动操作

    3. 没有定义析构操作

- `C++11` rules:

    1. **默认构造**：同 `C++98`，仅当类没有用户自定义沟站函数时才会生成

    2. **默认析构函数**：基本与 `C++98` 相同，只有基类析构为虚，生成的析构才是虚；有一点不同的是 `C++11` 默认生成的析构是 `noexcept` 的

    3. **默认拷贝构造**：行为同 `C++98`，拷贝非静态成员；生成规则不同，仅当无用户自定义拷贝构造时才生成，如果有用户定义的移动操作则不会生成，`98`中当用户自定义了拷贝赋值或者析构时生成默认拷贝构造的规则被废弃了

    4. **默认拷贝赋值**：与**默认拷贝构造**规则类似

    5. **默认移动构造**和**默认移动赋值**：移动非静态成员，仅当类没有用户定义的拷贝操作、移动操作、和析构时才会生成。

- ***NOTE***: 特殊成员函数模板不会阻止生成这些特殊的成员函数
