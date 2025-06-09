# Python与Git基础课程
## 1.变量、变量类型、作用域   

- 基本变量类型：`int`、`float`、`str`、`bool`、`list`、`tuple`、`dict`、`set`。
- 作用域：全局变量、局部变量，`global`和`nonlocal`关键字。
- 类型转换：如`int()`、`str()`。

## 2.运算符及表达式

- 算术运算符：`+`, , , `/`, `//`, `%`, `*`。
- 比较运算符：`==`, `!=`, `>`, `<`, `>=`, `<=`。
- 逻辑运算符：`and`, `or`, `not`。
- 位运算符：`&`, `|`, `^`, `<<`, `>>`。

## 3.语句：条件、循环、异常

- 条件语句：`if`, `elif`, `else`。
- 循环语句：`for`, `while`, `break`, `continue`。
- 异常处理：`try`, `except`, `finally`。

## 4.函数：定义、参数、匿名函数、高阶函数  

- 函数定义：`def`关键字，默认参数，可变参数（`args`, `*kwargs`）。
- 匿名函数：`lambda`。
- 高阶函数：接受函数作为参数或返回函数。

## 5.包和模块：定义模块、导入模块、使用模块、第三方模块  

- 模块：`import`语句，`from ... import ...`。
- 创建模块：一个`.py`文件。
- 包：包含`__init__.py`的文件夹。
- 第三方模块：如`requests`, `numpy`。

## 6.类和对象  

- 类定义：`class`关键字，属性和方法。
- 继承、多态、封装。
- 实例化对象。

## 7.装饰器  

- 装饰器本质：高阶函数，接受函数并返回新函数。
- 使用`@`语法。
- 带参数的装饰器。

## 8.文件操作  

- 读写文本文件：`open()`, `read()`, `write()`。
- 上下文管理器：`with`语句。
- 处理CSV、JSON文件。

# 学习内容1代码
```python
# 变量类型
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域
x = 10  # 全局变量
def my_function():
    y = 5  # 局部变量
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")
```

# 学习内容2代码
```python
# 算术运算
a = 10
b = 3
print(a + b)  # 13
print(a // b)  # 3（整除）
print(a ** b)  # 1000（幂）

# 逻辑运算
x = True
y = False
print(x and y)  # False
print(x or y)   # True

# 比较运算
print(a > b)  # True
```

# 学习内容3代码
```python
# 条件语句
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for i in range(5):
    if i == 3:
        continue
    print(i)

# 异常处理
try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")
```

# 学习内容4代码
```python
# 函数定义
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))  # Hello, Alice!
print(greet("Bob", "Hi"))  # Hi, Bob!

# 可变参数
def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1, 2, 3, 4))  # 10

# 匿名函数
double = lambda x: x * 2
print(double(5))  # 10

# 高阶函数
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 4))  # 16
```

# 学习内容5代码
```python
# 创建模块 mymodule.py
# mymodule.py
def say_hello():
    return "Hello from module!"

# 主程序
import mymodule
print(mymodule.say_hello())

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200

# 包使用示例
from mypackage import mymodule
```

# 学习内容6代码
```python
# 定义类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

# 继承
class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

# 使用
student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())  # I am Alice, 20 years old.
print(grad.introduce())     # I am Bob, a CS student.
```

# 学习内容7代码
```python
# 简单装饰器
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# 带参数的装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")
```

# 学习内容8代码
```python
# 写文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# 处理CSV
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
```