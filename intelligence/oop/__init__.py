"""Silverwing-ML OOP Educational Module."""

from intelligence.oop.basics import (
    Animal,
    Bird,
    Cat,
    Circle,
    Dog,
    Employee,
    Rectangle,
    Shape,
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
from intelligence.oop.composition import (
    File as CompFile,
)
from intelligence.oop.design_patterns import (
    AccessControlProxy,
    Coffee,
    Command,
    Database,
    Decorator,
    Espresso,
    EventEmitter,
    HeapSort,
    History,
    ImageProxy,
    LegacyPaymentProcessor,
    Logger,
    MacroCommand,
    MergeSort,
    MilkDecorator,
    PaymentProcessor,
    QueryBuilder,
    QuickSort,
    ShapeFactory,
    SingletonMeta,
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
    FileProcessor,
    Flyer,
    FlyingFish,
    JSONProcessor,
    Motorcycle,
    NotificationService,
    Pentagon,
    PushNotification,
    SMSNotification,
    Swimmer,
    TextProcessor,
    Truck,
    Vehicle,
)
from intelligence.oop.inheritance import (
    Circle as InhCircle,
)
from intelligence.oop.inheritance import (
    Rectangle as InhRectangle,
)
from intelligence.oop.inheritance import (
    Shape as InhShape,
)
from intelligence.oop.metaclasses import (
    AbstractBaseMeta,
    AutoRegisterMeta,
    FrozenInstanceMeta,
    ValidatedFieldsMeta,
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

__all__ = [
    "Animal", "Bird", "Cat", "Circle", "Dog", "Employee",
    "BasicBankAccount", "Rectangle", "Shape", "Triangle", "Vector2D",
    "BankAccount", "Person", "Product", "ReadOnlyAttribute", "Temperature",
    "ValidatedAttribute",
    "Car", "CSVProcessor", "InhCircle", "EmailNotification", "FlyingFish",
    "Flyer", "JSONProcessor", "Motorcycle", "NotificationService", "Pentagon",
    "PushNotification", "InhRectangle", "SMSNotification", "InhShape", "Swimmer",
    "TextProcessor", "Truck", "Vehicle", "FileProcessor",
    "Calculator", "CSVSerializer", "JSONSerializer", "Polynomial", "Serializer",
    "XMLSerializer", "draw", "process",
    "Command", "Database", "EventEmitter", "HeapSort", "History", "Logger",
    "MacroCommand", "MergeSort", "MilkDecorator", "QuickSort", "ShapeFactory",
    "Sorter", "SugarDecorator", "WhipDecorator", "Coffee", "Decorator",
    "Espresso", "AccessControlProxy", "ImageProxy", "LegacyPaymentProcessor",
    "PaymentProcessor", "VendingMachine", "QueryBuilder", "SingletonMeta",
    "Address", "CompCar", "Course", "Department", "Engine", "CompFile",
    "Folder", "FileSystem", "Observable", "Order", "OrderItem", "Student",
    "University", "User",
    "AutoRegisterMeta", "AbstractBaseMeta", "FrozenInstanceMeta",
    "MetaSingletonMeta", "ValidatedFieldsMeta",
]
