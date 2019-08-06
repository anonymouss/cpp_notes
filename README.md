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

    4. `e` 是 `T` 类型的 `prvalue`，推断结果为 `T`。*（`decltype((a + b))` 为 `T` 不是 `T &`，因为 `a + b` 为右值）*

