## Здесь буду писать как я сделал этот проект пошагова , и свои выводы.

1.Делать проект в Visual studio code реально легче и быстрее чем в jupyter notebook.
 Причины:
     1.Легче создавать Pipeline
     2.Можно 1 раз создать Pipeline и дальше просто дать значение Функции , 
     в Jupyter notebook нужно каждый раз менять Переменные.
     3.В Visual Studio code не нужно Перезапускать всё занова если что-то Сломалось.
     Например не нужно перезапускать ядро при скачивание Библиотеки как в Jupyter notebook. 
     4.Здесь можно выбрать Версию нужной Библиотеки. А в Juypter notebook так не получится
     5.Опять же все серезьные проект делаются в Visual Studio code с использованием:
        - Git, Docker, Dockerhub, Cloud Servides etc. 


Сгенерировать requirements.txt (в активированном .venv):
pip freeze > requirements.txt