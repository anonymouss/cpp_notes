# Kotlin

## Functions

### Infix Functions

```kotlin
fun main() {
    infix fun Int.times(str: String) = str.repeat(this)
    println(2 times "Bye")
    
    val pair = "Ferrari" to "Katrina"
    println(pair)
    
    infix fun String.onto(other: String) = Pair(this, other)
    val myPair = "McLaren" onto "Lucas"
    println(myPair)
    
   	class Person(val name: String) {
        val likedPeople = mutableListOf<Person>()
        infix fun likes(other: Person) {
            likedPeople.add(other)
        }
        
        fun say() {
            print(this.name + " likes ")
            for (people in likedPeople) {
                print(people.name + " ")
            }
            println()
        }
    }
    val sophia = Person("Sophia")
    val claudia = Person("Claudia")
    sophia likes claudia
    sophia.say()
}
```

### Operator Functions

```kotlin
fun main() {
    operator fun Int.times(str: String) = str.repeat(this)
    println(2 * "Bye")
    
    operator fun String.get(range: IntRange) = substring(range)
    val str = "Always forgive your enemies; nothing annoys them so much"
    println(str[0..14])
}
```

### Functions with `vararg` Parameters

```kotlin
fun printAll(vararg messages: String) {
    for (m in messages) println(m)
}
fun printAllWithPrefix(vararg messages: String, prefix: String) {
    for (m in messages) println(prefix + m)
}
fun log(vararg entries: String) {
    printAll(*entries)
}

fun main() {
    printAll("Hello", "Hallo", "Salut", "Hola", "你好")
    printAllWithPrefix(
    	"Hello", "Hallo", "Salut", "Hola", "你好",
        prefix = "Greeting: "
    )
}
```

## Null Safety

``` kotlin
fun main() {
    var neverNull: String = "This can't be null"
    // neverNull = null // Null can not be a value of a non-null type String
    
    var nullable: String? = "You can keep a null here"
    nullable = null
    
    var inferredNonNull = "The compiler assumes non-null"
    // inferredNonNull = null
    
    fun strLength(notNull: String): Int = notNull.length
    strLength(neverNull)
    // strLength(nullable) // Type mismatch: inferred type is Nothing? but String was expected
}
```

## Generic

```kotlin
// generic class
class MutableStack<E>(vararg items: E) {
    private val elements = items.toMutableList()
    fun push(element: E) = elements.add(element)
    fun peek(): E = elements.last()
    fun pop(): E = elements.removeAt(elements.size - 1)
    fun isEmpty() = elements.isEmpty()
    fun size() = elements.size
    override fun toString() = "MutableStack(${elements.joinToString()})"
}

// generic function
fun <E> mutableStackOf(vararg elements: E) = MutableStack(*elements)
```

## Iterators

```kotlin
class Animal(val name: String)
class Zoo(val animals: List<Animal>) {
    operator fun iterator(): Iterator<Animal> = animals.iterator()
}

fun main() {
    val zoo = Zoo(listOf(Animal("zebra"), Animal("lion")))
    for (animal in zoo) {
        println("Watch out, it's a ${animal.name}")
    }
}
```

## Special Classes

### Data Class

```kotlin
data class User(val name: String, val id: Int) {
    override fun equals(other: Any?) =
    	other is User && other.id == this.id
}

fun main() {
    val user = User("Alex", 1)
    println(user)
    
    val secondUser = User("Alex", 1)
    val thirdUser = User("Max", 2)
    println("user == secondUser: ${user == secondUser}")
    println("user == thirdUser: ${user == thirdUser}")
   	
    println(user.hashCode())
    println(secondUser.hashCode())
    println(thirdUser.hashCode())
    
    println(user.copy())
    println(user === user.copy())
    println(user.copy("Max"))
    println(user.copy(id = 3))
    
    println("name = ${user.component1()}")
    println("id = ${user.component2()}")
}
```

### Enum Classes

```kotlin
enum class State {
    IDLE, RUNNING, FINISHED
}

enum class Color(val rgb: Int) {
    RED(0xFF0000), GREEN(0x00FF00), BLUE(0x0000FF), YELLOW(0xFFFF00);
    
    fun containsRed() = (this.rgb and 0xFF0000 != 0)
}

fun main() {
    val state = State.RUNNING
    val message = when(state) {
        State.IDLE -> "it is idle"
        State.RUNNING -> "it's running"
        State.FINISHED -> "it's finished"
    }
    println(message)
    
    val red = Color.RED
    println(red)
    println(red.containsRed())
    println(Color.BLUE.containsRed())
    println(Color.YELLOW.containsRed())
}
```

### Sealed Classes

can only be subclassed from inside the same package where the sealed class is declared

```kotlin
sealed class Mammal(val name: String)
class Cat(val catName: String) : Mammal(catName)
class Human(val humanName: String, val job: String) : Mammal(humanName)

fun greetMammal(mammal: Mammal): String {
    when(mammal) {
        is Human -> return "Hello ${mammal.name}l You're working as a ${mammal.job}"
        is Cat -> return "Hello ${mammal.name}"
    }
}

fun main() {
    println(greetMammal(Cat("Snowy")))
}
```

### `object` Keyword

```kotlin
fun rentPrice(standardDays: Int, festivityDays: Int, specialDays: Int) {
    val dayRates = object {
        var standard: Int = 30 * standardDays
        var festivity: Int = 50 * festivityDays
        var special: Int = 100 * specialDays
    }
   	var total = dayRates.standard + dayRates.festivity + dayRates.special
    println("Total price: $$total")
}

object DoAuth {
    fun takeParams(username: String, password: String) {
        println("Input Auth parameters = $username:$password")
    }
}

class BigBen {
    companion object Bonger {
        fun getBongs(nTimes: Int) {
            for (i in 1..nTimes) {
                print("BONG ")
            }
        }
    }
}

fun main() {
    rentPrice(10, 2, 1)
    DoAuth.takeParams("foo", "qwerty")
    BigBen.getBongs(12)
}
```

## Functional

### High-Order-Functions

```kotlin
fun calculate(x: Int, y: Int, operation: (Int, Int) -> Int): Int {
    return operation(x, y)
}
fun sum(x: Int, y: Int) = x + y

fun operations(): (Int) -> Int {
    return ::square
}
fun square(x: Int) = x * x

fun main() {
    val sumRes = calculate(4, 5, ::sum)
    val mulRes = calculate(4, 5) {a, b -> a * b}
    println("Sum Result: ${sumRes}\nMul Result: ${mulRes}")
    
    val func = operations()
    println(func(2))
}
```

### Lambda Functions

```kotlin
fun main() {
    val upperCase1: (String) -> String = { str: String -> str.uppercase() }
    val upperCase2: (String) -> String = { str -> str.uppercase() }
    val upperCase3 = { str: String -> str.uppercase() }
    val upperCase4: (String) -> String = { it.uppercase() }
    val upperCase5: (String) -> String = String::uppercase
    
    println(upperCase1("hello"))
	println(upperCase2("hello"))
	println(upperCase3("hello"))
	println(upperCase4("hello"))
	println(upperCase5("hello"))
}
```

### Extension Functions and Properties

```kotlin
data class Item(val name: String, val price: Float)
data class Order(val items: Collection<Item>)

fun Order.maxPricedItemValue(): Float = this.items.maxByOrNull { it.price }?.price ?: 0F
fun Order.maxPricedItemName() = this.items.maxByOrNull { it.name }?.name ?: "NO_PRODUCT"

val Order.commaDelimitedItemNames: String
	get() = items.map { it.name }.joinToString()
    
fun <T> T?.nullSafeToString() = this?.toString() ?: "NULL"

fun main() {
    val order = Order(listOf(Item("Bread", 25.0F), Item("Wine", 29.0F), Item("Water", 12.0F)))
	println("Max price item name: ${order.maxPricedItemName()}")
	println("Max price item value: ${order.maxPricedItemValue()}")
    println("Items: ${order.commaDelimitedItemNames}")
    
    println(null.nullSafeToString())
    println("Kotlin".nullSafeToString())
}
```

## Collections

`List/MutableList`，`Set/MutableSet`，`Map/MutableMap`

```kotlin
const val POINTS_X_PASS: Int = 15
val EZPassAccounts: MutableMap<Int, Int> = mutableMapOf(1 to 100, 2 to 100, 3 to 100)
val EZPassReport: Map<Int, Int> = EZPassAccounts

fun updatePointsCredit(accountId: Int) {
    if (EZPassAccounts.containsKey(accountId)) {
        println("Updating $accountId...")
        EZPassAccounts[accountId] = EZPassAccounts.getValue(accountId) + POINTS_X_PASS
    } else {
        println("ERROR: Trying to update a non-existing account (id: $accountId)")
    }
}

fun accountsReport() {
    println("EZ-Pass report:")
    EZPassReport.forEach() {
        k, v -> println("ID $k: credit $v")
    }
}

fun main() {
    accountsReport()
    updatePointsCredit(1)
    updatePointsCredit(1)
    updatePointsCredit(5)
    accountsReport()
}
```

```kotlin
val numbers = listOf(1, -2, 3, -4, 5, -6)

// filter
val positives = numbers.filter { x -> x > 0 }
val evens = numbers.filter { it % 2 == 0 }

// map
val doubled = numbers.map { x -> x * 2}
val tripled = numbers.map { it * 3 }

// any, all, none
// returns true if contains at least one element matches given predicate
val anyNegative = numbers.any { it < 0 }
val anyGT6 = numbers.any { it > 6}
// returns true if all ...
val allEven = numbers.all { it % 2 == 0}
val allLess6 = numbers.all { it < 6 }
// returns true if no match ...
val allOdd = numbers.none { it %2 == 0 }
val allLess7 = numbers.none { it > 7 }

fun main() {
    println(positives)
    println(evens)
    println(doubled)
    println(tripled)
    println(anyNegative)
    println(anyGT6)
	println(allEven)
    println(allLess6)
    println(allOdd)
    println(allLess7)
}
```

```kotlin
data class Person(val name: String, val city: String, val phone: String)

fun main() {
    val people = listOf(
    	Person("John", "Boston", "+1-888-123456"),
        Person("Sarah", "Munich", "+49-777-789123"),
        Person("Svyatoslav", "Sait-Petersburg", "+7-999-456789"),
        Person("Vasilisa", "Sait-Petersburg", "+7-999-123456")
    )
    val phoneBook = people.associateBy { it.phone }
    val cityBook = people.associateBy(Person::phone, Person::city)
    val peopleCities = people.groupBy(Person::city, Person::name)
    val lastPersonCity = people.associateBy(Person::city, Person::name)
    
    println(phoneBook)
    println(cityBook)
    println(peopleCities)
    println(lastPersonCity)
}
```

```kotlin
fun main() {
	val numbers = listOf(1, -2, 3, -4, 5, -6)
	val evenOdd = numbers.partition { it % 2 == 0 }
	val (positives, negatives) = numbers.partition { it > 0 }
    println(evenOdd)
	println(positives)
	println(negatives)
}
```

```kotlin
fun main() {
	val fruitsBag = listOf("apple", "orange", "banana", "grapes")
	val clothesBag = listOf("shirts", "pants", "jeans")
	val cart = listOf(fruitsBag, clothesBag)
	val mapBag = cart.map { it }
	val flatMapBag = cart.flatMap {it}
	println(cart)
    println(mapBag)
    println(flatMapBag)
}
```

```kotlin
fun main() {
    val A = listOf("a", "b", "c")
    val B = listOf(1, 2, 3, 4)
    
    val resPairs = A zip B
    val resReduce = A.zip(B) { a, b -> "$a$b" }

	println(resPairs)
    println(resReduce)
}
```

## Scope Functions

`let`，`run`，`with`，`apply`，`also`

```kotlin
// let
fun printNonNull(str: String?) {
    println("Printing \"$str\":")
    str?.let {
        print("\t")
        customPrint(it)
    	println()
    }
}

fun printIfBothNonNull(strOne: String?, strTwo: String?) {
    strOne?.let {
        firstString -> strTwo?.let {
            secondString -> customPrint("$firstString: $secondString")
            println()
        }
    }
}

fun customPrint(str: String) = print(str.uppercase())

fun main() {
    val empty = "test".let {
        customPrint(it)
        it.isEmpty()
    }
    println(" is empty: $empty")
    
    printNonNull(null)
    printNonNull("my string")
    printIfBothNonNull("first", "second")
}
```

## Delegation Pattern

```kotlin
interface SoundBehavior {
    fun makeSound()
}

class ScreamBehavior(val n: String) : SoundBehavior {
    override fun makeSound() = println("${n.uppercase()}")
}
class RockAndRollBehavior(val n: String) : SoundBehavior {
    override fun makeSound() = println("I'm the King of Rock 'N' Roll: $n")
}

class TomAraya(n: String) : SoundBehavior by ScreamBehavior(n)
class ElvisPresley(n: String) : SoundBehavior by RockAndRollBehavior(n)

fun main() {
    val tomAraya = TomAraya("Thrash Metal")
    tomAraya.makeSound()
    val elvisPresley = ElvisPresley("Dacin' to the Jailhouse Rock.")
    elvisPresley.makeSound()
}
```








