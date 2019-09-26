# Head First: Design Patterns

## 设计原则

- 封装变化

- 针对接口编程

- 多用对象组合少用继承

- 为交互对象之间的松耦合设计而努力

- 开闭原则：对扩展开放，对修改关闭

- 依赖倒置：要依赖抽象，不用依赖具体类

- 最少知识原则：只和最亲密的朋友交流

- 好莱坞原则：不要找（调用）我，我会主动找（调用）你

- 单一责任：一个类应该只有一个引起变化的原因

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

## [装饰器与外观模式](./07_Adapter_and_Facade/)

![adapter](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Adapter_785-394.gif)

**适配器**：将一个类的接口，转换成客户期望的另一个接口。适配器让原本接口不兼容的类可以合作无间

![facade](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Facade_701-268.gif)

**外观模式**：提供了一个统一的接口，用来访问子系统中的一群接口。外观定义了一个高层接口，让子系统更容易使用。

## [模板方法](./08_Template_Method/)

![template method](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/TemplateMethod_452-300.gif)

**模板方法**：在一个方法中定义一个算法的骨架，而将一些步骤延迟到子类中。模板方法使得子类可以在不改变算法结构的情况下，重新定义算法中的某些步骤。

## [迭代器与组合模式](./09_Iterator_and_Composite/)

![iterator](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Iterator_486-242.gif)

**迭代器模式**：提供一个方法，顺序访问一个聚合对象中的各个元素而又不暴露其内部实现。

![composite](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Composite_713-363.gif)

**组合模式**：允许你将对象组合成树形结构来表现“整体/部分”层次结构。组合能让客户以一致的方式处理个别对象以及对象组合。

## [状态模式](./10_State_Pattern/)

![state](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/State_554-244.gif)

**状态模式**：允许对象在内部状态改变时改变它的行为，看起来对象好像修改了它的类。

*eg.* Android [`ACodec`](https://android.googlesource.com/platform/frameworks/av/+/refs/heads/master/media/libstagefright/ACodec.cpp)

## [代理模式](./11_Proxy_Pattern/)

![proxy](https://sites.cs.ucsb.edu/~mikec/cs48/misc/Design_Class_Diagrams_files/Proxy_660-272.gif)

**代理模式**：为另一个对象提供一个替身或占位符以控制这个对象的访问

*eg.* Android [`Binder`](https://github.com/anonymouss/android-ipc-demo/tree/master/binder-demo)
