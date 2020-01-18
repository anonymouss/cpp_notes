# Notes of <C++ Templates: The Complete Guide, 2 Edition>

## 基础

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

## 深入

### 深入了解基础

#### 参数化声明

- C++目前支持4种基本的模板：类模板，函数模板，变量模板和别名模板。

- union模板

- 默认调用参数

```cpp
template <typename T>
void fill(Array<T> &, const T & = T());

struct Value {
    explicit Value(int);
};

void init(Array<Value> &array) {
    Value zero(0);
    fill(array, zero);  // OK
    fill(array);        // ERROR, Value no default ctor
}
```

- 类模板的非模板成员

##### 虚成员函数

-  成员函数模板不可以被定义为虚函数，因为虚函数表是固定大小的，但是模板函数在整个程序翻译完成之前无法确定数量。

```cpp
template <typename T>
struct Dynamic {
    virtual ~Dynamic();             // OK

    template <typename T2>
    virtual void copy(const T2 &);  // ERROR!
};
```

##### 模板的链接

- 一般模板具有外部链接，`static`修饰或者匿名命名空间的模板具有内部链接，匿名类的模板没有链接

```cpp
template <typename T>   // refers to the same entity as a declaration of the
void external();        // same name (and scope) in other file


template <typename T>   // unrelated to a template with the same name in
static void internal(); // another file

template <typename T>   // redeclaration of the previous declaration
static void internal();


namespace {
    template <typename>     // also unrelated to a template with same name
    void otherInternal();   // in another file, even one that similarly appears
                            // in an unnamed namespace
}

namespace {
    template <typename>     // redeclaration of the previous template declaration
    void otherInternal();
}


struct {
    template <typename T>   // no linkage: canot be redeclared
    void f(T) {}
} x;
```

##### 基本模板

#### 模板形参

##### 类型形参

##### 非类型形参

必须是在编译或者链接时可以确定的常量值

- 整型或枚举类型

- 指针类型

- 指向成员的指针

- 左值引用（包括指向对象和指向函数的引用）

- `std::nullptr_t`

- 任何含有 `auto` 或者 `decltype(auto)` 的类型

其他类型都不可以（浮点类型将来可能会支持）

##### 模板的模板形参

模板的模板参数是类或者别名模板的占位符。其声明和类模板很类似，但是不能使用`struct`和`union`关键字

```cpp
template <
    template <typename X> class C   // OK
> void f(C<int> *p);

template <
    template <typename X> struct C  // ERROR: struct is not valid here
> void f(C<int> *p);

template <
    template <typename X> union C   // ERROR: union is not valid here
> void f(C<int> *p);

template <
    template <typename X> typename C    // OK, sicne C++17
> void f(C<int> *p);
```

- 模板的模板参数的模板参数只能被它自己访问到，外层模板无法访问

- 因此模板的模板参数的模板参数名，如果后面用不掉，可以省略不写

##### 模板形参包（变长参数模板）

##### 默认模板参数

- 类/变量/别名模板只有当后续模板参数有了默认参数时才可，函数模板无此限制

- 默认模板参数**不可**重复（重复定义）

- **不可**以在特化模板中定义参数（必须在基本模板）

```cpp
template <typename T> class C;
template <typename T = int> class C<T*>;    // ERROR
```

- 参数包（变长）**不可**以有默认参数

- 类模板成员在类外定义**不可**有默认参数

- 友元类模板声明**不可**有默认参数，除非它是定义，并且在别处没有声明

```cpp
struct S {
    template <typename = void> friend void f();     //ERROR! not permitted
    template <typename = void> friend void g() {}   // OK
    template <typename> void g();                   // ERROR! alreay defined with a default arg.
                                                    // forbide declare anywhere else
};
```

#### 模板实参

- 模板的模板实参必须是一个类模板或者别名模板，它本身也有模板参数，（C++17前）该参数必须严格匹配传入的模板的模板参数，且默认参数不被考虑

```cpp
// prior C++ 17
#include <list>
/**
 * template <typename T, typename Allocator = allocator<T>> class list;
 */

template <typename T1, typename T2, template<typename> class Cont>
struct Rel {
    Cont<T1> c1;
    Cont<T2> c2;
};

Rel<int, double, std::list> rel; // error! std::list requires two template args
```

- 可变参数可以突破这个限制 [`test_tmpl_args.h`](./book_samples/01_basics/include/test_tmpl_args.h)

- **包扩展**

- 友元

### 模板中的名称

#### 名称分类

- 受限名

- 依赖名

#### 名称查找

- `ADL`（参数依赖查找）：对非受限名函数查找时，除了在当前作用域往上查找外，还会在**参数作用域**查找（但是会**忽略**参数作用域内的`using指示符`，即忽略`using`引入的作用域）。

- inject name

- 待决名（依赖名）- 加 `typename`

- 依赖类型模板 - 加 `template`

#### 派生和类模板

- 非依赖型基类（无需知道模板实参就可以完全确定基类）

- **注意**：在派生类中查找非受限名时，会先查找非依赖基类，然后才查找模板参数列表

```cpp
template <typename X>
struct Base {
    int basefield;
    typedef int T;  // NOTE
};

template <typename T>
struct Derived : public Base<double> {  // no dependent base class
    void f() { basefield = 7; }
    T strange;    // NOTE: here T is Base<doule>::T --> int
};

void g(Derived<int *> &d, int *p) {
    d.strange = p;  // ERROR! d.strange is int type, p is int *
}
```

- 依赖型基类（基类类型与模板实参有关）

- 标准规定看到非依赖名要立即查找，因而此时无法查找依赖基类，因为它要到模板实例化阶段才能确定

```cpp
template <typename T>
struct DD : public Base<T> {    // dependent base class
    void f() { basefield = 0; } // ERROR! undefined! because can't lookup Base<T>
};

template <>
struct Base<bool> {  // explicitly specialization
    enum { basefield = 42 };
};

void g(DD<bool> &d) { d.f(); } // oops!

// fix 1
template <typename T>
struct DD1 : public Base<T> {    // dependent base class
    void f() { this->basefield = 0; } // ERROR!
};

// fix 2
template <typename T>
struct DD2 : public Base<T> {
    void f() { Base<T>::basefield = 0; } // NOTE: virtual func is prohibited in this case
};

// fix 3
template <typename T>
struct DD3 : public Base<T> {
    using Base<T>::basefield;
    void f() { basefield = 0; }
};
```

### 实例化

#### On-Demand 实例化（隐式实例化）

#### 延迟实例化

- 模板实例化时只实例化必须的部分

```cpp
template <typename T> class Q {
    using Type = typename T::Type;
};

Q<int> *p = nullptr;    // OK! 编译完全没有问题，尽管 int::Type 是错的
```

#### C++ 实例化模型

##### 两步查找

- step 1. 模板解析；step 2. 模板实例化

#### 编译器中的一些实现方案

##### 贪婪实例化

##### 查询实例化

##### 迭代实例化

#### 显式实例化

