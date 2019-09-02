# Head First: Design Patterns

## 设计原则

- 封装变化

- 针对接口编程

- 多用对象组合少用继承

- 为交互对象之间的松耦合设计而努力

- 开闭原则：对扩展开放，对修改关闭

- 依赖倒置：要依赖抽象，不用依赖具体类

## [策略模式](./01_Strategy/)

![strategy](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Stategy_801-243.gif)

对象有某个行为，但在不同的场景中，该行为有不同的算法实现。

所以**策略模式**：定义了算法族，分别封装起来，让他们之间可以互相替换，此模式让算法的变化独立于使用算法的客户。

## [观察者模式](./02_Observer/)

![observer](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Observer_833-283.gif)

**观察者模式**：在对象之间定义一对多的依赖，这样一来当一个对象改版状态，依赖它的对象全都会收到通知并自动更新

## [装饰者模式](./03_Decorator/)

![decorator](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Decorator_723-422.gif)

**装饰者模式**：动态地将责任附加到对象上。想要扩展对象，装饰者提供有别于继承的另一种选择。

## [工厂模式](./04_Factory/)

### 工厂方法模式

![factory method](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/FactoryMethod_848-296.gif)

**工厂方法模式**：定义了一个创建对象的接口，但由其子类决定要实例化的对象类是哪一个。工厂方法让类把实例化推迟到了子类

![abstract factory](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/AbstractFactory_741-283.gif)

**抽象工厂模式**：提供一个接口，用于创建相关或依赖对象的家族，而不需要明确指定具体类

简单工厂（静态工厂方法）：一个工厂类对应多种对象（工厂责任过重）

工厂方法：多个子工厂类各自对应多种对象（是简单工厂的扩展）

抽象工厂：各个工厂对应产品对象族（更加抽象），通过工厂组合实例化具体的产品

## [单例模式](./05_Singleton/)

![singleton](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Singleton_459-182.gif)

**单例模式**：确保一个类只有一个实例，并提供一个全局的访问点

## [命令模式](./06_Command/)

![command](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Command_854-316.gif)

**命令模式**：将请求封装成对象，以便使用不同的请求、队列或者日志来参数化其他对象。命令模式也支持可撤销的操作

*eg.* Android [`NuPlayer::Action`](https://android.googlesource.com/platform/frameworks/av/+/refs/heads/master/media/libmediaplayerservice/nuplayer/NuPlayer.cpp)
