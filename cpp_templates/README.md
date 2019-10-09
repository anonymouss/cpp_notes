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

浮点数和类对象是不可以的。

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
|:-----------------------:|:-------------------------------------------:|
| `(... op pack)`         | `(((pack1 op pack2) op pack3)... op packN)` |
| `(pack op ...)`         | `(pack1 op (... (packN-1 op packN)))`       |
| `(init op ... op pack)` | `(((init op pack1) op pack2)... op packN)`  |
| `(pack op ... op init)` | `(pack1 op (... (packN op init)))`          |

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
