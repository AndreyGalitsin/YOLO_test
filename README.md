# YOLO test

## Задача
- Есть модель YOLO, обученная на n из N целевых классов.
- Есть датасет с размеченными данными для n классов
- Необходимо разработать систему полуавтоматичесой разметки, которая позволит находить и размечать недостающие классы на изображениях, на которых модель не обучалась

## Пайплайн
1. Для недостающих классов готовим набор изображений (5-6 шт)
2. Готовим embedding matrix для этих объектов 
3. Прогоняем все изображения из датасета через обученную модель yolo с порогом 0.85 и находим уверенные классы, которые модель умеет размечать / подгружаем файл с существующей разметкой в coco аннотации 
4. Маскируем найденные bbox (зануляем пиксели)
5. Маскированные изображения прогоняем через fastSAM и определяем bbox каких-то иных объектов, которые присутствуют на изображении 
6. Для каждого найденного bbox в fastSAM делаем embedding 
7. Сравниваем его через cosin sim с embedding matrix 
8. Если значение сравнения больше 0.85, добавляем объект в разметку 
9. Если значение 0.7-0.85, добавляем объект в список, который нужно отдать на проверку человеку 
10. Человек через Label Studio проходится по этим изображением, размечает их вручную и данные добавляются в разметку 

PS.: Для разметки используется coco аннотация

## Запуск
1. В папку checkpoints добавить веса моделей FastSAM-x.pt и yolo_marking.pt
2. В папке data/new_classes создать папки с названием новых классов и добавить в них по 5-6 примеров
3. Запустить скрипт self_labeling.py. Указать параметр images_folder_path - путь, где лежат папки изображениями для разметки
4. self_labeling.py сгенерирует в указанной папке два файла: final_coco_annotation.json - уверенная разметка изображений и coco_to_check_annotation.json - разметка, которую нужно проверить человеку
5. cocoviewer.py позволяет просматривать разметку в формате coco. Для запуска нужно указать путь к файлу с coco аннотацией

Краткий анализ тестового датасета и инференса обученной модели:

| № | Классы | YOLO обучена на классе | Класс есть в датасете | Класс хорошо распознаётся моделью | Комментарий                                                 |
|---|--------|------------------------|-----------------------------------|-----------------------------------|-------------------------------------------------------------|
| 1 | Alenka milk chocolate | ✅ |  ❌  | -                                 |                                                             |
| 2 | Baltic 0° Original | ✅ |  ❌| -                                 |                                                             |
| 3 | Baltic 0° Raspberry | ✅ | ❌ | -                                 |                                                             |
| 4 | Borjomi Cherry and Pomegranate | ✅ |❌  | -                                 |                                                             |
| 5 | Borjomi Mandarin | ✅ |❌  | -                                 |                                                             |
| 6 | Borjomi Pear | ✅ |❌  | -                                 |                                                             |
| 7 | Cone Forest water | ✅ |  ✅| ✅  |Модель распознаёт хорошо                                                             |
| 8 | Extra Granola fruits, berries and nuts | ✅ | ❌  | -                                 |                                                             |
| 9 | Extra Granola milk chocolate | ✅ | ❌  | -                                 |                                                             |
| 10 | Head&Shoulders shampoo | ✅ | ❌  | -                                 |                                                             |
| 11 | Jubilee biscuit icing | ✅ |❌  | -                                 |                                                             |
| 12 | Limon Fresh pear | ✅ |❌  | -                                 | Модель называет этим классом сок сады придонья              |
| 13 | Makfa spaghetti №16 | ✅ |✅| ✅                                 | Модель распознаёт хорошо                                                            |
| 14 | Mr.Ricco spicy ketchup | ✅ |❌  | -                                 |                                                             |
| 15 | Pedigree dog food | ✅ |❌  | -                                 |                                                             |
| 16 | Purina One cat treat salmon | ✅ |❌  | -                                 |                                                             |
| 17 | Rexona Men deodarant | ✅ |✅| ✅                                 | Модель распознаёт хорошо                                    |
| 18 | Ritter Sport cookie with nuts chocolate | ✅ |❌  | -                                 |                                                             |
| 19 | Samokat canned peaches | ✅ |✅| ✅                                 | Модель иногда путает с Lays Stax и Samokat buckwheat flakes |
| 20 | Samokat feather pasta | ✅ |❌  | -                                 |                                                             |
| 21 | Samokat sunflower seed oil | ✅ |✅| ✅                                 | Модель распознаёт хорошо                                    |
| 22 | Snickers bar | ✅ |❌  | -                                 |                                                             |
| 23 | Splat proffesional biocalcium tooth paste | ✅ |❌  | -                                 |                                                             |
| 24 | TiTBiT dog treat | ✅ |❌  | -                                 | Модель называет этим классом сок сады придонья              |
| 25 | AOS detergent | ❌ |❌  | -                                 |                                                             |
| 26 | Adrenalin Extra | ❌ |❌  | -                                 |                                                             |
| 27 | Adrenalin Original | ❌ |❌  | -                                 |                                                             |
| 28 | Adrenalin Zero | ❌ |❌  | -                                 |                                                             |
| 29 | Agusha apple juice | ❌ |❌  | -                                 |                                                             |
| 30 | Agusha apple-briar juice | ❌ |❌  | -                                 |                                                             |
| 31 | Agusha apple-cherry juice | ❌ |❌  | -                                 |                                                             |
| 32 | Bambolina apple juice | ❌ |❌  | -                                 |                                                             |
| 33 | Bambolina apple-briar juice | ❌ |❌  | -                                 |                                                             |
| 34 | Bambolina multifruit juice | ❌ |❌  | -                                 |                                                             |
| 35 | Belvita morning cereal biscuits | ❌ |❌  | -                                 |                                                             |
| 36 | BigBon Wok spicy beef | ❌ |❌  | -                                 |                                                             |
| 37 | Choco Pie Original | ❌ |❌  | -                                 |                                                             |
| 38 | Chudo vanilla ice cream milkshake | ❌ |❌  | -                                 |                                                             |
| 39 | Clean Cat sponges | ❌ |❌  | -                                 |                                                             |
| 40 | Diseta foil 10m (wide) | ❌ |❌  | -                                 |                                                             |
| 41 | Domik V Derevne cream 10% | ❌ |❌  | -                                 |                                                             |
| 42 | Domik V Derevne cream 20% | ❌ |❌  | -                                 |                                                             |
| 43 | Foil 10m | ❌ |❌  | -                                 |                                                             |
| 44 | Fruto Nanny banana milkshake | ❌ |❌  | -                                 |                                                             |
| 45 | Fruto Nanny berry morsel | ❌ |❌  | -                                 |                                                             |
| 46 | Fruto Nanny compote apple, raisins and apricots | ❌ |❌  | -                                 |                                                             |
| 47 | Fruto Nanny multifruit juice | ❌ |❌  | -                                 |                                                             |
| 48 | Gardens Pridonya apple juice | ❌ |✅| ❌                                 | Модель иногда ложно называет Limon Fresh pear               |
| 49 | Gardens Pridonya apple-cherry juice | ❌ |✅| ❌                                 | Модель иногда ложно называет Limon Fresh pear               |
| 50 | Greenfield black tea | ❌ |❌  | -                                 |                                                             |
| 51 | J7 orange juice | ❌ |❌  | -                                 |                                                             |
| 52 | J7 pineapple juice | ❌ |✅| ❌                                 | Не распознаётся моделью                                     |
| 53 | Kruazett Original Crisps | ❌ |❌  | -                                 |                                                             |
| 54 | Lay's Stax spicy paprika | ❌ |✅| ❌                                 | Модель ложно называет Samokat canned peaches                |
| 55 | Lay's crab | ❌ |✅| ❌                                 | Модель ложно называет Jubilee biscuit icing                 |
| 56 | Limon Fresh berries | ❌ |❌  | -                                 |                                                             |
| 57 | Limon Fresh mint | ❌ |❌  | -                                 |                                                             |
| 58 | Martin seeds | ❌ |❌  | -                                 |                                                             |
| 59 | Medovik Cake Day | ❌ |❌  | -                                 |                                                             |
| 60 | Milk chocolate M&M's | ❌ |❌  | -                                 |                                                             |
| 61 | Nemoloko oatmeal chocolate milk | ❌ |❌  | -                                 |                                                             |
| 62 | Nemoloko oatmeal vanilla milk | ❌ |❌  | -                                 |                                                             |
| 63 | Parmalat milk 1.8% | ❌ |❌  | -                                 |                                                             |
| 64 | Parmalat milk 3.5% | ❌ |❌  | -                                 |                                                             |
| 65 | Peanut M&M's | ❌ |❌  | -                                 |                                                             |
| 66 | Pemolux Soda 7 | ❌ |❌  | -                                 |                                                             |
| 67 | Purina Pro Plan cat food | ❌ |❌  | -                                 |                                                             |
| 68 | Ritter Sport dunkle voll-nuss chocolate | ❌ |❌  | -                                 |                                                             |
| 69 | Rollton noodles with beef | ❌ |✅| ❌                                 | Не распознаётся моделью                                     |
| 70 | RothFront candy bar | ❌ |❌  | -                                 |                                                             |
| 71 | Samokat Nicaragua ground coffee | ❌ |❌  | -                                 |                                                             |
| 72 | Samokat baby disposable nappies | ❌ |❌  | -                                 |                                                             |
| 73 | Samokat baby wet wipes | ❌ |❌  | -                                 |                                                             |
| 74 | Samokat buckwheat | ❌ |❌  | -                                 |                                                             |
| 75 | Samokat buckwheat flakes | ❌ | ✅| ❌                                 | Модель называет этот класс Samokat canned peaches           |
| 76 | Samokat cotton disks | ❌ |❌  | -                                 |                                                             |
| 77 | Samokat couscous | ❌ | ❌  | -                                 |                                                             |
| 78 | Samokat grapefruit juice | ❌ | ❌  | -                                 |                                                             |
| 79 | Samokat mango oolong tea | ❌ |❌  | -                                 |                                                             |
| 80 | Samokat napkins | ❌ |❌  | -                                 |                                                             |
| 81 | Samokat parboiled rice | ❌ | ❌  | -                                 |                                                             |
| 82 | Samokat sea salt with garlic | ❌ |❌  | -                                 |                                                             |
| 83 | Semushka cashews | ❌ | ❌  | -                                 |                                                             |
| 84 | Semushka pistachio | ❌ |❌  | -                                 |                                                             |
| 85 | Splat proffesional sensitiv tooth paste | ❌ |❌  | -                                 |                                                             |
| 86 | Splat proffesional whitening plus tooth paste | ❌ | ❌  | -                                 |                                                             |
| 87 | Tuc sour cream and onion | ❌ |❌  | -                                 |                                                             |
| 88 | Vifresh cranberry morsel | ❌ | ❌  | -                                 |                                                             |
| 89 | Zelenika yam chips | ❌ |❌  | -                                 |                                                             |
