# MyFirstDataProject2

Настоящая работа посвящена созданию модели машинного обучения и сервиса, определяющих генетические расстройства, в рамках прохождения курса "My First Data Project 2: от идеи к продукту" (https://ods.ai/tracks/mfdp2).

## О геномах и генетике 

Контекст

С самого зарождения человеческой жизни на земле численность мирового населения стремительно росла. 
По оценкам, в 1800 году население составляло 1 миллиард человек. К началу двадцатого века эта цифра достигла нового максимума - 
6 миллиардов человек. Изо дня в день в мире прибавляется 227 000 человек; по прогнозам, к концу 21 века население планеты может 
превысить 11 миллиардов.
Согласно отчетам, в результате неустойчивого роста населения и отсутствия доступа к надлежащему медицинскому обслуживанию, 
пище и жилью увеличилось число заболеваний, связанных с генетическими расстройствами. Наследственные заболевания становятся все более 
распространенными из-за отсутствия понимания необходимости генетического тестирования. Часто дети умирают в результате этих 
заболеваний, поэтому генетическое тестирование во время беременности имеет решающее значение.

Задача

Предоставлен набор данных, содержащий медицинскую информацию о детях с генетическими нарушениями. Задача - предсказать следующее:

1. Генетическое расстройств;
2. Подкласс расстройств

Материал для задачи взят с: https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge.
В материалах оригинального задания представлены 3 документа: train.csv, test.csv и sample_submission.csv.
Однако в рамках настоящего задания будет использоваться только train.csv для возможности определния качества подобранной ML-модели на тестовой выборке, а также предсказываться будет только класс генетических расстройств.
