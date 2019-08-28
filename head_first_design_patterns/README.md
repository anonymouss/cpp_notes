# Head First: Design Patterns

## 设计原则

- 封装变化

- 针对接口编程

- 多用组合少用继承

- 为交互对象之间的松耦合设计而努力

- 开闭原则：对扩展开放，对修改关闭

## 策略模式

![strategy](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Stategy_801-243.gif)

对象有某个行为，但在不同的场景中，该行为有不同的算法实现。

所以**策略模式**：定义了算法族，分别封装起来，让他们之间可以互相替换，此模式让算法的变化独立于使用算法的客户。
