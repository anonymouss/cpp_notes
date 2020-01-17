# Notes of <C++ Templates: The Complete Guide, 2 Edition>

## The Basics

### 函数模板

- 模板并非是编译成一个可以处理任意类型的实体函数，而是为每个用到的类型生成不同的实体函数。

#### 两阶段查找

    1. 在定义时，模板并不实例化，只对独立的（与模板参数无关）的部分进行检查。例如：语法错误，未定义的变量名（非待决名），静态断言等

    2. 在实例化时，对待决名进行检查（dependent name）

***两阶段查找是有坑的***：

http://blog.llvm.org/2009/12/dreaded-two-phase-name-lookup.html

https://blog.codingnow.com/2010/01/cpp_template.html

#### 模板类型推断

see < Effective Modern C++ >

### 类模板

在类的内部，成员函数不必显式的使用`<T>`，因为他们和类使用同样的模板参数（不同时需要指出）。但是在类的外部，则需要指明。

```cpp
template <typename T>
void Stack<T>::pop() { // implementation }
```

类模板的定义或声明必须是**全局**的！discussed in section 12.1

注意，模板都是在使用到它时才实例化的（except显式实例化），包括类模板的成员函数。所以，即使类模板某些成员函数定义了某些操作是我们实例化时使用的模板参数类不支持的操作也没有关系。因为只有你真正去调用这个成员函数时，它才会被实例化，错误才会发生。

#### `Concepts`

这是一个C++目前还没有支持的特性，用来限定模板可接受的类型。目前的workaround可以使用`std::enable_if`等。

#### 友元

友元函数它不是当前类的成员，它是一个普通函数。有三种方法来定义：

1. 最简单的，直接声明并且定义在类内部

2. 在类内部声明，类外定义。这时，接受的类实例化参数要显式声明，并且和类的参数区分开来防止覆盖

3. 在类外声明定义，在类内为具体类型实例化

#### 类模板特化与偏特化

注意如果有超过一个偏特化对参数可以完美匹配，会发生歧义

#### 默认模板参数

#### 模板别名

### 非类型模板参数

限制：通常，只可以是整型常量值（包括枚举）、指向对象/函数/成员的指针、对象或函数的左值引用或`std::nullptr_t`

**浮点数和类对象是不可以的。**

传递模板参数给指针或引用是，对象不可以是字符串字面值、临时对象或者内部数据成员与其他子类型。虽然在C++17开始放宽了这些限制

    • In C++03, 必须有外部链接

    • In C++11/14, 有外部链接内部链接都可以

    • In C++17，即使没有链接也可以

```cpp
extern char const s03[] = "hi";      // external linkage
char const s11[] = "hi";             // internal linkage

int main() {
    Message<s03> m03;                // OK (all versions)
    Message<s11> m11;                // OK since C++11
    static char const s17[] = "hi";  // no linkage
    Message<s17> m17;                // OK since C++17
}
```

### 可变参数模板

#### 折叠表达式（C++17）

```c++
template <typename... T>
auto foldSum(T... S) {
    return (... + s);                // (((s1) + s2) + s3) ...
}
```

|     Fold Expression     |                 Evaluation                  |
| :---------------------: | :-----------------------------------------: |
|     `(... op pack)`     | `(((pack1 op pack2) op pack3)... op packN)` |
|     `(pack op ...)`     |    `(pack1 op (... (packN-1 op packN)))`    |
| `(init op ... op pack)` | `(((init op pack1) op pack2)... op packN)`  |
| `(pack op ... op init)` |     `(pack1 op (... (packN op init)))`      |

### Ticky Basics

- 待决名不可缺少 `typename`

- 避免无法默认初始化的模板参数，要手动零值初始化，like `T x{};`，模板类请定义默认构造函数

- 模板类继承的情况下，要通过`this->`或`Base<T>::`之类显式调用成员函数

```cpp
template <typename T>
class Base {
public:
    void bar();
};

template <typename T>
class Derived : public Base<T> {
public:
    void foo() {
        bar();      // calls external bar() or error, Base<T>::bar() is nerver considered
    }
};
```

- 注意原生数组和字符串字面值模板参数退化的情况

- `template` 为待决名消歧义

```cpp
template <unsigned long N>
void printBitset(const std::bitset<N> &bs) {
    // distinguish from less-than token
    std::cout << bs.template to_string<char, std::char_traits<char>, std::allocator<char>>();
}
```

- 变量模板

```cpp
template <typename T>
constexpr T pi{3.1415926535897932385};

// usage
std::cout << pi<double> << std::endl;
std::cout << pi<float> << std::endl;
```

### 移动语义和`enable_if<>`

#### 完美转发

#### 特殊成员函数模板

- 注意数组（字符串字面常量）退化的情形（字符数组，`std::string`）

- 注意构造函数模板可能会优先匹配，隐藏掉预定义的构造函数（如拷贝构造函数，因为它的`const`会不如模板匹配的非`const`）

```cpp
class Person {
public:
    template <typename STR>
    Person(STR &&n) : name(std::forward<STR>(n)) {} // tmpl

    Person(cosnt Person &p) : name(p.name) {} // copy-ctor
    Person(Person &&p) : name(std::move(p.name)) {} // move-ctor

private:
    std::string name;
};

// usage
Person p1;
Person p2(p2);
// tmpl generate --> Person(Person &p);
// copy ctor     --> Person(const Person &p);
// tmpl matches best
// std::enable_if<> could fix this (or concept in future)
```

#### `std::enable_if<>`禁用模板

- https://zh.cppreference.com/w/cpp/types/enable_if

- 拷贝/移动/赋值这些特殊函数是不会被模板函数disable的，一个人tricky的方法是：定义拷贝构造的参数为`const volatile`并且把它`delete`

```cpp
class C {
public:
    // user-define the predefined copy constructor as deleted
    // (with conversion to volatile to enable better matches)
    C(C const volatile&) = delete;

    // implement copy constructor template with better match:
    template<typename T>
    C (T const&) {
        std::cout << "tmpl copy constructor\n";
    }
};

// Usage
C x;
C y{x}; // uses the member template
```

#### Concept

- `std::enable_if<>`用起来很笨拙，需要语言层面的支持（`concept`)，looking forward to C++20... :)

### 传值还是引用？

#### 传值

- 值传递时参数会被拷贝，对于类来说拷贝构造成本可能很高，但是有的时候会被编译器优化（RVO、copy elision）

- 退化情形，如模板推导时忽略掉`cv`限定符、字符串字面值退化为指针，数组退化为指针等

#### 传引用

- `std::ref()` & `std::cref()`

- 字符串字面值和数组的特殊模板函数实现

```cpp
// only valid for arrays
template <typename T, std::size_t L1, std::size_t L2>
void foo(T (&arg1)[L1], T (&arg2)[L2]) { /* ... */ }

// use type traits to detect whether an array(or a pointer) is passed
template <typename T, typename = std::enable_if_t<std::is_array_v<T>>>
void foo(T &&arg1, T &&arg2) { /* ... */ }
```

#### 建议

- 一般情况下声明参数为值传递，对于大类型（copy expensive），调用者可以使用`std::ref()`或`std::cref()`来传递以避免拷贝

- 如果参数需要作为允许修改的返回值使用，使用非常量引用（可以考虑使用type traits拒绝接受常量引用实参）

- 如果参数需要转发，使用通用引用（可以考虑使用`std::decay<>`或`std::common_type<>`解决字符串字面值和裸数组推导为不同类型的问题/长度不一致）

- 如果性能要求很高，使用常量引用

- 如果是高手，自己决定。。。记住，不要过分的追求通用（generic）

### 编译期编程

- `template`

- `constexpr` 函数（不一定编译器求值，取决于调用者的情况）

- 通过偏特化选择执行的路径

- SFINAE(out)

- `if constexpr(...)` < *since C++17* >

### 模板实践

#### The Inclusion Model

- 模板的实现和定义通常不要分离（分离在单独的`.h`和`.cpp`文件），这可能导致编译器不知道需要实例化模板而导致链接错误

- 将模板定义与实现现在一个文件也有缺点，就是会导致编译时间变长，但是现在没有更好的办法。未来可能有`modules`

#### 模板与内联

- 模板是否会被内联替换完全取决于编译器，不过可以指定编译器属性如`noinline`或者`always_inline`

#### 预编译头（vendor specific impl）

