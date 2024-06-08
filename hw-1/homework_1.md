# <center> Homework 1 </center>

# 0. Где мы сейчас?
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" width="150">

[Jupyter Notebook](https://jupyter.org/) - интерактивная среда для запуска программного кода в браузере. Удобный инструмент для анализа данных, который используется многими специалистами по data science. Позволяет выполнять отдельные ячейки с кодом, а не всю программу сразу, что очень удобно при знакомстве с данными.

# 1. Python
> Python - это свободный интерпретируемый объектно-ориентированный расширяемый встраиваемый язык программирования очень высокого уровня 

>*(Г.Россум, Ф.Л.Дж.Дрейк, Д.С.Откидач "Язык программирования Python").*

А если без цитат, то питон - это просто очень крутой язык, созданный в 1991 году [Великодушным пожизненным диктатором](https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D0%BB%D0%B8%D0%BA%D0%BE%D0%B4%D1%83%D1%88%D0%BD%D1%8B%D0%B9_%D0%BF%D0%BE%D0%B6%D0%B8%D0%B7%D0%BD%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B4%D0%B8%D0%BA%D1%82%D0%B0%D1%82%D0%BE%D1%80) Гвидо ван Россумом и названный в честь любимого им шоу [Monty Python's Flying Circus](https://en.wikipedia.org/wiki/Monty_Python%27s_Flying_Circus)

<img src="https://advancelocal-adapter-image-uploads.s3.amazonaws.com/image.oregonlive.com/home/olive-media/width2048/img/tv/photo/2018/10/10/montycastjpg-7ef393e2355a42aa.jpg" width="300">

# 2. Anaconda

<img src="https://cdn-images-1.medium.com/max/1600/0*MVkCW8_Bmj-nuAnI.png" width="300">

[Сборка Anaconda](https://www.anaconda.com/products/individual) включает очень много полезных библиотек для анализа данных. 

Среди наиболее популярных библиотек:
 - <a href="http://numpy.org">Numpy</a> - это один из основных пакетов для математических вычислений. Он содержит средства для работы с многомерными массивами и высокоуровневыми математическими функциями
 - <a href="https://www.scipy.org/">SciPy</a> - научные вычисления. Методы оптимизации, интегрирования, модули обработки сигналов и изображений, статистика, линейная алгебра, сплайны, кластеризация и многое другое
 -  <a href="http://pandas.pydata.org/">Pandas</a> - основная библиотека для обработки и анализа данных. Предназначена для данных разной природы - матричных, панельных данных, временных рядов. Претендует на звание самого мощного и гибкого средства для анализа данных с открытым исходным кодом
 - <a href="http://scikit-learn.org/stable/">Scikit-learn</a> - реализация очень многих методов машинного обучения с отличной документацией. 
 - <a href="http://http://matplotlib.org/">matplotlib</a> - хорошая библиотека для визуализации данных

## Задача 1
$N$ хоббитов делят $K$ кусков эльфийского хлеба поровну, не делящийся нацело остаток остается в корзинке у Сэма. Напишите функцию, которая принимает на вход параметры $N$ и $K$ и возвращает два числа: $x$ - cколько кусков эльфиского хлеба достанется каждому хоббиту, и $y$ - сколько кусков остаётся в корзинке.


```python
def share_bread(N, K):
    # Вычисляем количество кусков хлеба на одного хоббита
    x = K // N
    # Вычисляем остаток
    y = K % N
    
    return x, y

assert share_bread(N=3, K=14) == (4, 2)
```

## Задача 2

В копях Мории хоббиты нашли стену, на которой высечены разные натуральные числа. Согласно древним сказаниям, это даты сражений. Хоббиты знают, что сражения происходили только по високосным годам. Помогите хоббитам определить, является ли год с данным числом датой великого сражения. Если это так, то верните строку "YOU SHALL PASS", иначе верните "YOU SHALL NOT PASS". Напомним, что в соответствии с хоббитским календарем, год является високосным, если его номер кратен 4, но не кратен 100, а также если он кратен 400.


```python
def leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        return "YOU SHALL PASS"
    else:
        return "YOU SHALL NOT PASS"

assert leap_year(5) == 'YOU SHALL NOT PASS'
```

## Задача 3
<img src="http://i0.kym-cdn.com/photos/images/original/001/187/255/5e9.gif" width=300>
Для могущественного магического ритуала Гендальфу необходимо быстро подсчитывать площадь своего амулета, который умеет менять размеры. Известно, что амулет имеет форму треугольника и Гендальф знает длину каждой из сторон. Напишите функцию, которая считает площадь амулета по трем сторонам.  

Подсказка: используйте формулу Герона
$$ S = \sqrt{p(p-a)(p-b)(p-c)} $$
$$ p = \frac{a + b + c}{2}$$



```python
def amulet_area(a, b, c):
    # Вычисляем полупериметр
    p = (a + b + c) / 2
    # Вычисляем площадь по формуле Герона
    s = (p  *  (p - a)  *  (p - b)  *  (p - c))  **  0.5
    return s


assert amulet_area(3, 4, 5) == 6
```

## Задача 4

Хоббиты собираются пешком идти до Мордора и им нужно подсчитать расстояние, которое им предстоит пройти. Хоббиты смогли вспомнить сразу несколько метрик расстояния: евклидово, манхэттена и косинусное, так что ваша задача - напистаь функцию под каждую из них. Важное условие - используйте только базовые функции numpy для решения.


* Евклидово расстояние
$$ d(a, b) = \sqrt{\sum_i (a_i - b_i)^2} $$
* Расстояние Манхэттена
$$ d(a, b) = \sum_i |a_i - b_i| $$
* Косинусное расстояние
$$ d(a, b) = 1 - \frac{a^\top b}{||a||_2\cdot||b||_2}$$


```python
%pip install numpy
import numpy as np
```


```python
def cal_euclidean(a, b):    
    return np.linalg.norm(a - b)

def cal_manhattan(a, b):    
    return np.sum(np.abs(a - b))


def cal_cosine(a, b):    
    # Нормализация векторов для получения единичных векторов
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    a_unit = a / norm_a if norm_a != 0 else a
    b_unit = b / norm_b if norm_b != 0 else b
    
    # Вычисление косинусного расстояния
    return 1 - np.dot(a_unit, b_unit)

```


```python
a = np.random.randint(-10, 10, size=10)
b = np.random.randint(-10, 10, size=10)
print(cal_euclidean(a, b))
print(cal_manhattan(a, b))
print(cal_cosine(a, b))
```

## Задача 5

Ну и напоследок, еще немного практики numpy, без которой не обходится ни один хоббит.


Создайте случайный array (`np.random.rand()`) длинной 100. Преобразуйте его так, чтобы
* Максимальный элемент(ы) был равен 1
* Минимальный элемент(ы) был равен 0
* Остальные элементы в итнтервале от 0 до 1 остаются прежними


```python
# Создаем случайный массив длиной 100
my_array = np.random.rand(100)
# Находим индекс максимального элемента
max_index = np.argmax(my_array)
# Устанавливаем максимальный элемент равным 1
my_array[max_index] = 1
# Находим индекс минимального элемента
min_index = np.argmin(my_array)
# Устанавливаем минимальный элемент равным 0
my_array[min_index] = 0
# Ограничиваем остальные элементы значением от 0 до 1
arr = np.clip(my_array, a_min=0, a_max=1)


print(np.max(my_array), np.min(my_array))
print(my_array)
```

Создайте array размером $5 \times 6$ с целыми числами в интервале [0,50]. Напечатайте колонку, которая содержит максимальный элемент полученной матрицы 


```python
my_array = np.random.random((5, 6))
max_index = np.unravel_index(my_array.argmax(), my_array.shape)
selected_column = my_array[max_index[0]]
print('Shape: ',my_array.shape)
print('Array')
print(my_array)
print(selected_column)
```

Напишите функцию, которая принимает на вохд матрицу (array) X и возвращает все её уникальные строки в виде новой матрицы.


```python
def get_unique_rows(X):
    unique_rows, unique_indices = np.unique(X, axis=0, return_index=True)    

    return unique_rows
```


```python
X = np.random.randint(4, 6, size=(10,3))
print(X)
```


```python
get_unique_rows(X)
```
