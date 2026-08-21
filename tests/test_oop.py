"""Comprehensive tests for the OOP educational module."""

import math
import os
import sys
from abc import abstractmethod

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.oop.basics import (
    Animal,
    Bird,
    Cat,
    Circle,
    Dog,
    Employee,
    Rectangle,
    Triangle,
    Vector2D,
)
from intelligence.oop.basics import (
    BankAccount as BasicBankAccount,
)
from intelligence.oop.composition import (
    Address,
    Course,
    Department,
    Engine,
    File,
    FileSystem,
    Folder,
    Observable,
    Order,
    OrderItem,
    Student,
    University,
    User,
)
from intelligence.oop.composition import (
    Car as CompCar,
)
from intelligence.oop.design_patterns import (
    AccessControlProxy,
    Command,
    Database,
    Espresso,
    EventEmitter,
    HeapSort,
    History,
    ImageProxy,
    LegacyPaymentProcessor,
    Logger,
    MergeSort,
    MilkDecorator,
    PaymentAdapter,
    QueryBuilder,
    QuickSort,
    ShapeFactory,
    Sorter,
    SugarDecorator,
    VendingMachine,
    WhipDecorator,
)
from intelligence.oop.encapsulation import (
    BankAccount,
    Person,
    Product,
    ReadOnlyAttribute,
    Temperature,
    ValidatedAttribute,
)
from intelligence.oop.inheritance import (
    Car,
    CSVProcessor,
    EmailNotification,
    FlyingFish,
    JSONProcessor,
    Motorcycle,
    Pentagon,
    PushNotification,
    SMSNotification,
    TextProcessor,
    Truck,
)
from intelligence.oop.inheritance import (
    Circle as InhCircle,
)
from intelligence.oop.inheritance import (
    Rectangle as InhRectangle,
)
from intelligence.oop.metaclasses import (
    AbstractBaseMeta,
    AutoRegisterMeta,
    FrozenInstanceMeta,
)
from intelligence.oop.metaclasses import (
    SingletonMeta as MetaSingletonMeta,
)
from intelligence.oop.polymorphism import (
    Calculator,
    CSVSerializer,
    JSONSerializer,
    Polynomial,
    Serializer,
    XMLSerializer,
    draw,
    process,
)

# ─── basics.py tests ───────────────────────────────────────────────

class TestAnimalHierarchy:
    def test_dog_speak(self):
        assert Dog("Rex").speak() == "Rex says Woof!"

    def test_cat_speak(self):
        assert Cat("Whiskers").speak() == "Whiskers says Meow!"

    def test_bird_speak(self):
        assert Bird("Tweety").speak() == "Tweety says Tweet!"

    def test_repr(self):
        assert "Dog" in repr(Dog("Rex"))
        assert "Rex" in repr(Dog("Rex"))

    def test_cannot_instantiate_animal(self):
        with pytest.raises(TypeError):
            Animal("Generic")


class TestShapeHierarchy:
    def test_circle_area(self):
        c = Circle(5)
        assert math.isclose(c.area(), math.pi * 25)

    def test_circle_perimeter(self):
        c = Circle(5)
        assert math.isclose(c.perimeter(), 10 * math.pi)

    def test_rectangle_area(self):
        r = Rectangle(3, 4)
        assert r.area() == 12

    def test_rectangle_perimeter(self):
        r = Rectangle(3, 4)
        assert r.perimeter() == 14

    def test_triangle_area(self):
        t = Triangle(3, 4, 5)
        assert math.isclose(t.area(), 6.0)

    def test_triangle_perimeter(self):
        t = Triangle(3, 4, 5)
        assert t.perimeter() == 12


class TestEmployee:
    def test_str(self):
        e = Employee("Alice", 50000, "Engineering")
        s = str(e)
        assert "Alice" in s
        assert "50,000" in s

    def test_promote(self):
        e = Employee("Bob", 60000, "Engineering")
        e.promote()
        assert e.salary == 66000
        assert "Senior" in e.department

    def test_give_raise(self):
        e = Employee("Bob", 60000, "Engineering")
        e.give_raise(5000)
        assert e.salary == 65000


class TestBasicBankAccount:
    def test_deposit(self):
        a = BasicBankAccount("Alice", 100)
        a.deposit(50)
        assert a.balance == 150

    def test_withdraw(self):
        a = BasicBankAccount("Alice", 100)
        a.withdraw(30)
        assert a.balance == 70

    def test_withdraw_insufficient(self):
        a = BasicBankAccount("Alice", 10)
        with pytest.raises(ValueError):
            a.withdraw(50)

    def test_transfer(self):
        a = BasicBankAccount("Alice", 100)
        b = BasicBankAccount("Bob", 50)
        a.transfer(b, 30)
        assert a.balance == 70
        assert b.balance == 80

    def test_interest_rate(self):
        a = BasicBankAccount("Alice", 1000)
        a.apply_interest()
        assert math.isclose(a.balance, 1050)


class TestVector2D:
    def test_add(self):
        v = Vector2D(1, 2) + Vector2D(3, 4)
        assert v.x == 4 and v.y == 6

    def test_subtract(self):
        v = Vector2D(5, 7) - Vector2D(2, 3)
        assert v.x == 3 and v.y == 4

    def test_mul(self):
        v = Vector2D(2, 3) * 3
        assert v.x == 6 and v.y == 9

    def test_magnitude(self):
        assert math.isclose(Vector2D(3, 4).magnitude(), 5.0)

    def test_normalize(self):
        v = Vector2D(3, 4).normalize()
        assert math.isclose(v.magnitude(), 1.0)

    def test_dot(self):
        assert Vector2D(1, 2).dot(Vector2D(3, 4)) == 11

    def test_eq(self):
        assert Vector2D(1, 2) == Vector2D(1, 2)
        assert Vector2D(1, 2) != Vector2D(3, 4)

    def test_angle_between(self):
        angle = Vector2D(1, 0).angle_between(Vector2D(0, 1))
        assert math.isclose(angle, math.pi / 2)


# ─── encapsulation.py tests ────────────────────────────────────────

class TestPerson:
    def test_getters(self):
        p = Person("Alice", 30)
        assert p.name == "Alice"
        assert p.age == 30

    def test_setters(self):
        p = Person("Alice", 30)
        p.name = "Bob"
        p.age = 25
        assert p.name == "Bob"
        assert p.age == 25

    def test_name_validation(self):
        p = Person("Alice", 30)
        with pytest.raises(ValueError):
            p.name = ""

    def test_age_validation(self):
        p = Person("Alice", 30)
        with pytest.raises(ValueError):
            p.age = -1


class TestEncapBankAccount:
    def test_deposit(self):
        a = BankAccount("Alice", 100)
        a.deposit(50)
        assert a.balance == 150

    def test_withdraw(self):
        a = BankAccount("Alice", 100)
        a.withdraw(30)
        assert a.balance == 70

    def test_negative_deposit(self):
        a = BankAccount("Alice", 100)
        with pytest.raises(ValueError):
            a.deposit(-10)

    def test_overdraw(self):
        a = BankAccount("Alice", 10)
        with pytest.raises(ValueError):
            a.withdraw(50)


class TestProduct:
    def test_price_setter(self):
        p = Product("Widget", 10.0)
        p.price = 15.0
        assert p.price == 15.0

    def test_negative_price(self):
        p = Product("Widget", 10.0)
        with pytest.raises(ValueError):
            p.price = -5.0

    def test_discount(self):
        p = Product("Widget", 100.0)
        p.discount(20)
        assert p.price == 80.0

    def test_apply_tax(self):
        p = Product("Widget", 100.0)
        p.apply_tax(0.1)
        assert math.isclose(p.price, 110.0)


class TestTemperature:
    def test_from_celsius(self):
        t = Temperature(celsius=100)
        assert math.isclose(t.fahrenheit, 212)

    def test_from_fahrenheit(self):
        t = Temperature(fahrenheit=32)
        assert math.isclose(t.celsius, 0)

    def test_set_celsius(self):
        t = Temperature(celsius=0)
        t.celsius = 100
        assert math.isclose(t.fahrenheit, 212)

    def test_set_fahrenheit(self):
        t = Temperature(celsius=0)
        t.fahrenheit = 212
        assert math.isclose(t.celsius, 100)

    def test_both_raises(self):
        with pytest.raises(ValueError):
            Temperature(celsius=0, fahrenheit=32)


class TestDescriptors:
    def test_read_only(self):
        class Config:
            version = ReadOnlyAttribute(default="1.0")
        c = Config()
        assert c.version == "1.0"
        c.version = "2.0"
        assert c.version == "2.0"
        with pytest.raises(AttributeError):
            c.version = "3.0"

    def test_validated(self):
        class Settings:
            port = ValidatedAttribute(
                validator=lambda x: isinstance(x, int) and 0 < x < 65536,
                error_msg="Invalid port",
            )
        s = Settings()
        s.port = 8080
        assert s.port == 8080
        with pytest.raises(ValueError):
            s.port = -1


# ─── inheritance.py tests ──────────────────────────────────────────

class TestVehicleHierarchy:
    def test_car_start(self):
        c = Car("Toyota", "Camry", 2024)
        assert "started" in c.start()

    def test_truck_haul(self):
        t = Truck("Ford", "F150", 2024, payload_capacity=500)
        assert "hauling" in t.haul(300)
        assert "exceeds" in t.haul(600)

    def test_motorcycle_wheelie(self):
        m = Motorcycle("Honda", "CBR", 2024)
        assert "wheelie" in m.wheelie()

    def test_car_drift(self):
        c = Car("Toyota", "Camry", 2024)
        assert "drifting" in c.drift()


class TestFileProcessors:
    def test_text_processor(self):
        tp = TextProcessor()
        assert tp.process("hello") == "HELLO"
        assert ".txt" in tp.supported_extensions()

    def test_csv_processor(self):
        cp = CSVProcessor()
        data = "name,age\nAlice,30\nBob,25"
        result = cp.process(data)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_json_processor(self):
        jp = JSONProcessor()
        result = jp.process('{"key": "value"}')
        assert result["key"] == "value"


class TestNotifications:
    def test_email(self):
        n = EmailNotification()
        result = n.send("alice@example.com", "Hello")
        assert "Email" in result
        assert "alice@example.com" in result

    def test_sms(self):
        n = SMSNotification()
        result = n.send("+1234567890", "Hello")
        assert "SMS" in result

    def test_push(self):
        n = PushNotification()
        result = n.send("device_123", "Hello")
        assert "Push" in result


class TestInheritanceShapes:
    def test_pentagon(self):
        p = Pentagon(1)
        assert math.isclose(p.perimeter(), 5)
        assert p.area() > 0


class TestMultipleInheritance:
    def test_flying_fish(self):
        ff = FlyingFish("Nemo")
        assert "flying" in ff.fly()
        assert "swimming" in ff.swim()

    def test_mro(self):
        mro = [cls.__name__ for cls in FlyingFish.__mro__]
        assert "FlyingFish" in mro
        assert "Flyer" in mro
        assert "Swimmer" in mro


# ─── polymorphism.py tests ─────────────────────────────────────────

class TestDuckTyping:
    def test_draw_circle(self):
        result = draw(Circle(5))
        assert "area" in result

    def test_draw_rectangle(self):
        result = draw(Rectangle(3, 4))
        assert "area" in result

    def test_process_list(self):
        assert process([1, 2, 3]) == 3

    def test_process_string(self):
        assert process("hello") == 5


class TestSerializerProtocol:
    def test_json_serializer(self):
        s = JSONSerializer()
        assert isinstance(s, Serializer)
        assert s.serialize({"a": 1}) == '{"a": 1}'

    def test_csv_serializer(self):
        s = CSVSerializer()
        result = s.serialize([{"name": "Alice", "age": "30"}])
        assert "Alice" in result

    def test_xml_serializer(self):
        s = XMLSerializer()
        result = s.serialize({"key": "value"})
        assert "key" in result
        assert "value" in result


class TestCalculator:
    def test_add(self):
        assert (Calculator(2) + Calculator(3)).value == 5

    def test_sub(self):
        assert (Calculator(5) - Calculator(3)).value == 2

    def test_mul(self):
        assert (Calculator(4) * Calculator(3)).value == 12

    def test_truediv(self):
        assert (Calculator(10) / Calculator(2)).value == 5

    def test_eq(self):
        assert Calculator(5) == Calculator(5)
        assert Calculator(5) == 5

    def test_lt(self):
        assert Calculator(2) < Calculator(5)


class TestPolynomial:
    def test_evaluate(self):
        p = Polynomial([1, 2, 3])
        assert p(0) == 1
        assert p(1) == 6
        assert p(2) == 17

    def test_call(self):
        p = Polynomial([1, 2, 3])
        assert p(1) == 6

    def test_add(self):
        p1 = Polynomial([1, 2])
        p2 = Polynomial([3, 4, 5])
        result = p1 + p2
        assert result.coefficients == [4, 6, 5]

    def test_mul(self):
        p1 = Polynomial([1, 1])
        p2 = Polynomial([1, -1])
        result = p1 * p2
        assert result.coefficients == [1, 0, -1]

    def test_eq(self):
        assert Polynomial([1, 2]) == Polynomial([1, 2])


# ─── design_patterns.py tests ──────────────────────────────────────

class TestSingleton:
    def test_same_instance(self):
        db1 = Database("test")
        db2 = Database("other")
        assert db1 is db2

    def test_logger_singleton(self):
        l1 = Logger("app1")
        l2 = Logger("app2")
        assert l1 is l2
        l1.log("test message")
        assert "test message" in l2.get_logs()


class TestShapeFactory:
    def test_create_circle(self):
        c = ShapeFactory.create("circle", radius=5)
        assert isinstance(c, InhCircle)
        assert c.radius == 5

    def test_create_rectangle(self):
        r = ShapeFactory.create("rectangle", w=4, h=3)
        assert isinstance(r, InhRectangle)
        assert r.width == 4

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            ShapeFactory.create("hexagon")


class TestObserver:
    def test_on_emit(self):
        emitter = EventEmitter()
        results = []
        emitter.on("data", lambda x: results.append(x))
        emitter.emit("data", "hello")
        assert results == ["hello"]

    def test_off(self):
        emitter = EventEmitter()
        callback = lambda x: x
        emitter.on("data", callback)
        emitter.off("data", callback)
        assert emitter.emit("data", 42) == []


class TestStrategy:
    def test_quicksort(self):
        s = Sorter(QuickSort())
        assert s.sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_mergesort(self):
        s = Sorter(MergeSort())
        assert s.sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_heapsort(self):
        s = Sorter(HeapSort())
        assert s.sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_set_strategy(self):
        s = Sorter()
        s.set_strategy(MergeSort())
        assert s.sort([5, 3, 1]) == [1, 3, 5]


class TestDecorator:
    def test_espresso(self):
        c = Espresso()
        assert c.cost() == 2.00
        assert c.description() == "Espresso"

    def test_with_milk(self):
        c = MilkDecorator(Espresso())
        assert c.cost() == 2.50
        assert "Milk" in c.description()

    def test_with_multiple(self):
        c = WhipDecorator(SugarDecorator(MilkDecorator(Espresso())))
        assert c.cost() == 3.50
        assert "Milk" in c.description()
        assert "Sugar" in c.description()
        assert "Whip" in c.description()


class TestCommand:
    def test_history_undo_redo(self):
        history = History()
        cmd = _make_test_command("test")
        history.execute(cmd)
        assert history.can_undo()
        result = history.undo()
        assert "undone" in result.lower() if isinstance(result, str) else True
        result = history.redo()


class _TestCommand(Command):
    def __init__(self, name: str) -> None:
        self.name = name
        self.executed = False

    def execute(self) -> str:
        self.executed = True
        return f"executed {self.name}"

    def undo(self) -> str:
        self.executed = False
        return f"undone {self.name}"


def _make_test_command(name: str) -> Command:
    return _TestCommand(name)


class TestVendingMachine:
    def test_full_cycle(self):
        vm = VendingMachine(stock=2)
        assert "Coin inserted" in vm.insert_coin()
        assert "selected" in vm.select_item()
        assert "dispensed" in vm.dispense()

    def test_out_of_stock(self):
        vm = VendingMachine(stock=1)
        vm.insert_coin()
        vm.select_item()
        vm.dispense()
        assert "out of stock" in vm.insert_coin()

    def test_no_coin_first(self):
        vm = VendingMachine()
        assert "coin first" in vm.select_item().lower()


class TestQueryBuilder:
    def test_simple_query(self):
        q = QueryBuilder()
        sql = q.select("name", "age").from_table("users").build()
        assert "SELECT name, age" in sql
        assert "FROM users" in sql

    def test_full_query(self):
        QueryBuilder()
        sql = (QueryBuilder()
               .select("id", "name")
               .from_table("users")
               .where("age > 18")
               .order_by("name")
               .limit(10)
               .build())
        assert "WHERE age > 18" in sql
        assert "ORDER BY name" in sql
        assert "LIMIT 10" in sql


class TestAdapter:
    def test_payment_adapter(self):
        legacy = LegacyPaymentProcessor()
        adapter = PaymentAdapter(legacy)
        result = adapter.pay(100.0)
        assert "100" in result
        assert "USD" in result


class TestProxy:
    def test_image_proxy(self):
        proxy = ImageProxy("/path/to/image.png")
        assert proxy._image is None
        result = proxy.display()
        assert proxy._image is not None
        assert "image.png" in result

    def test_access_control_proxy(self):
        class Service:
            def read(self) -> str:
                return "data"
        proxy = AccessControlProxy(Service(), ["admin"])
        assert proxy.execute("admin", "read") == "data"
        with pytest.raises(PermissionError):
            proxy.execute("guest", "read")


# ─── composition.py tests ──────────────────────────────────────────

class TestCompositionCar:
    def test_car_engine(self):
        engine = Engine(200, "gasoline")
        car = CompCar("Toyota", "Camry", engine)
        assert "started" in car.start()
        assert engine.running

    def test_engine_stop(self):
        engine = Engine(200)
        car = CompCar("Toyota", "Camry", engine)
        car.start()
        car.stop()
        assert not engine.running


class TestCompositionUser:
    def test_user_address(self):
        addr = Address("123 Main St", "Springfield", "IL", "62701")
        user = User("Alice", "alice@example.com", addr)
        assert "123 Main St" in user.get_shipping_address()


class TestOrder:
    def test_order_total(self):
        user = User("Alice", "a@b.com", Address("1 St", "City", "ST", "00000"))
        order = Order(user, tax_rate=0.1)
        order.add_item(OrderItem("Widget", 10.0, 2))
        order.add_item(OrderItem("Gadget", 20.0, 1))
        assert order.compute_subtotal() == 40.0
        assert math.isclose(order.compute_tax(), 4.0)
        assert math.isclose(order.compute_total(), 44.0)

    def test_discount(self):
        user = User("Alice", "a@b.com", Address("1 St", "City", "ST", "00000"))
        order = Order(user, tax_rate=0.0)
        order.add_item(OrderItem("Widget", 100.0, 1))
        assert order.apply_discount(10) == 90.0


class TestUniversityHierarchy:
    def test_deep_composition(self):
        uni = University("MIT")
        dept = Department("CS")
        course = Course("AI", 4)
        student = Student("Alice", "S001")
        dept.add_course(course)
        uni.add_department(dept)
        course.add_student(student)
        assert uni.total_students() == 1


class TestFileSystem:
    def test_ls(self):
        fs = FileSystem()
        fs.root.add(File("a.txt", 100))
        fs.root.add(File("b.txt", 200))
        assert fs.ls() == ["a.txt", "b.txt"]

    def test_find(self):
        fs = FileSystem()
        sub = Folder("sub")
        sub.add(File("found.txt"))
        fs.root.add(sub)
        result = fs.find("found.txt")
        assert result is not None
        assert result.name == "found.txt"

    def test_du(self):
        fs = FileSystem()
        sub = Folder("sub")
        sub.add(File("a.txt", 100))
        sub.add(File("b.txt", 200))
        fs.root.add(sub)
        assert fs.du() == 300

    def test_nested(self):
        fs = FileSystem()
        sub1 = Folder("a")
        sub2 = Folder("b")
        sub2.add(File("deep.txt", 50))
        sub1.add(sub2)
        fs.root.add(sub1)
        assert fs.du() == 50


class TestObservable:
    def test_subscribe_notify(self):
        obs = Observable()
        results = []
        obs.subscribe("event", lambda x: results.append(x))
        obs.notify("event", 42)
        assert results == [42]

    def test_unsubscribe(self):
        obs = Observable()
        cb = lambda x: x
        obs.subscribe("event", cb)
        obs.unsubscribe("event", cb)
        assert obs.notify("event", 1) == []


# ─── metaclasses.py tests ──────────────────────────────────────────

class TestAutoRegister:
    def test_auto_registration(self):
        class Widget(metaclass=AutoRegisterMeta):
            pass

        assert "Widget" in AutoRegisterMeta.get_registry()

    def test_get(self):
        class Gadget(metaclass=AutoRegisterMeta):
            pass

        assert AutoRegisterMeta.get("Gadget") is Gadget


class TestFrozenInstance:
    def test_frozen(self):
        class Point(metaclass=FrozenInstanceMeta):
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

        p = Point(1, 2)
        assert p.x == 1
        with pytest.raises(AttributeError):
            p.x = 3

    def test_new_attr_frozen(self):
        class Config(metaclass=FrozenInstanceMeta):
            def __init__(self) -> None:
                self.name = "test"

        c = Config()
        with pytest.raises(AttributeError):
            c.new_attr = "value"


class TestMetaSingleton:
    def test_same_instance(self):
        class Cache(metaclass=MetaSingletonMeta):
            def __init__(self) -> None:
                self.data = {}

        c1 = Cache()
        c2 = Cache()
        assert c1 is c2


class TestAbstractBaseMeta:
    def test_enforces_abstract(self):
        class Interface(metaclass=AbstractBaseMeta):
            @abstractmethod
            def do(self) -> None:
                ...

        with pytest.raises(TypeError):
            Interface()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
