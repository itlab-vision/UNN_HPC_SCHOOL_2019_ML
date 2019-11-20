# Практика 3. Повышение производительности модели детектирования объектов. Применение Calibration Tool в составе OpenVINO toolkit

## Цели

__Цель данной работы__ - изучить подходы для оптимизации производительности
глубоких моделей и научиться использовать инструмент Calibration Tool в составе
Intel Distribution of OpenVINO Toolkit для преобразования моделей в формат INT8.

## Задачи

__Основные задачи:__

 1. Изучить инструмент Calibration Tool.
 1. Подготовить данные и скрипты для калибровки модели `mobilenet-ssd`.
 1. Откалибровать модель `mobilenet-ssd`.
 1. Измерить скорость работы оригинальной модели и модели, калиброванной в INT8.
  
## Общая последовательность действий

 1. Установить зависимости, необходимые для работы Calibration Tool.
 1. Скачать и распаковать набор данных, который будет использован для калибрации.
 1. Разработать скрипты для выполнения калибровки сети.
 1. Выполнить калибровку.
 1. Измерить производительность модели в форматах FP32 и INT8 для синхронноого
    режима вывода и сравнить их.

## Детальная инструкция по выполнению работы

Данная инструкция основана на оригинальной документации, доступной по [ссылке][calibration-docs].
     
 1. Скачать тренировочный набор данных Pascal VOC и распакуйте в вашу рабочую директорию.         
    Для калибрации модели `mobilenet-ssd` воспользуйтесь датасетом PASCAL-VOC 2007 
    trainval dataset, на котором обучалась данная модель. Поскольку на 
    официальном сайте загрузка недоступна, можно воспользоваться 
    [зеркалом][pascal-voc-2007].
 
 1. Активировать виртуальную среду Python, созданную на [практике 1][practice_1].
    
 1. Перейти в рабочую папку на компьютере и установить зависимости для Calibration Tool.
 
    ```bash
    $ cd C:\UNN_HPC_SCHOOL_2019_OPENVINO
    $ python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\open_model_zoo\tools\accuracy_checker\setup.py" install
    ```

 1. Создать файл формата `.pickle`, который содержит набор изображений и 
    выходов, необходимых для калибровки. Для выполнения калибровки
    достаточно использовать небольшую часть набора данных 
    на несколько тысяч или даже сотен изображений, поскольку модель не переобучается,
    а осуществляется корректировка весов.
    
    ```bash
    $  python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\tools\accuracy_checker_tool\convert_annotation.py" voc_detection \
      --annotations_dir C:\dev\openvino_calibration\VOCdevkit\VOC2007\Annotations \
      --imageset_file C:\dev\openvino_calibration\VOCdevkit\VOC2007\ImageSets\Main\train.txt \
      --images_dir C:\dev\openvino_calibration\VOCdevkit\VOC2007\JPEGImages \
      -a C:\dev\openvino_calibration\train_images.pickle \
      -m C:\dev\openvino_calibration\train_images.meta -ss 500
    ```
    
    Подробное описание доступно по [ссылке][convert_annotation]
 
 1. Создать файл `sdd_calibrate.yml`, который будет содержать файл эксперимента калибрации:
 
    ```
    ssd_calibrate.yml
    _________________________________________________________________
    models:
      - name: mobilenet-ssd
        launchers:
          - framework: dlsdk
            device: CPU
        model: mobilenet-ssd.xml
            weights: mobilenet-ssd.bin
            adapter: ssd
            cpu_extensions: C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll
    
        datasets:
          - name: VOC2007
            data_source: C:\dev\openvino_calibration\VOCdevkit\VOC2007\JPEGImages
            annotation: C:\dev\openvino_calibration\train_images.pickle
            dataset_meta: C:\dev\openvino_calibration\train_images.meta
            preprocessing:
              - type: resize
                size: 300
              - type: normalization
                mean: 104, 117, 123
            postprocessing:
              - type: resize_prediction_boxes
            metrics:
              - type: map
                integral: 11point
                ignore_difficult: True
                presenter: print_scalar
    _________________________________________________________________
    ```
    
    Данный файл содержит в себе описание задачи, которую решает модель,
    путь до файлов модели, путь до файла с описанием набора данных `pickle`.
    
    Подробное описание доступно по [ссылке][adapters].
    
 1. Запустить калибровку модели.
 
    ```bash
    $ python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\tools\calibration_tool\calibrate.py" \
    -c C:\UNN_HPC_SCHOOL_2019_OPENVINO\config.yml \
    -m C:\UNN_HPC_SCHOOL_2019_OPENVINO\models \
    -s C:\UNN_HPC_SCHOOL_2019_OPENVINO \
    -M "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer" \
    --annotations C:\UNN_HPC_SCHOOL_2019_OPENVINO
    ```
 
 1. Запустить получившуюся модель в формате INT8 в вашем демо-приложении
    с предыдущей практики.
 
 1. Добавьте в ваше демо-приложение измерение скорости работы модели.
    Для измерения производительности вывода глубоких моделей в синхронном режиме
    необходимо использовать следующие показатели:

      - __Латентность (latency)__ - среднее время прямого прохода по модели.
        Для определения латентности вывод выполняется многократно (количество
        запусков = количество итераций). На каждой итерации измеряется время вывода
        (время работы функции `output = self.exec_net.infer(inputs = {input_blob: blob})`),
        и формируется набор времен. Для этого набора необходимо определить математическое
        ожидание (`time_mean = numpy.mean(times)`) и среднеквадратическое отклонение
        (`time_std = numpy.std(times)`). Далее следует отбросить времена, выходящие
        за пределы трех среднеквадратических отклонений. По оставшемуся набору необходимо
        вычислить среднее время.
      - __FPS__ - количество изображений в секунду, которое может обработать модель.
        Для вычисления FPS необходимо разделить общее количество обработанных 
        кадров на общее время их обработки. 

 1. Сравните скорость работы FP32, FP16 и INT8 варианта модели mobileNet-ssd.
 
<!-- LINKS -->
[calibration-docs]: https://docs.openvinotoolkit.org/2019_R3.1/_inference_engine_tools_calibration_tool_README.html
[pascal-voc-2007]: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
[practice_1]: practice1.md
[practice_2]: practice2.md
[adapters]: http://docs.openvinotoolkit.org/2019_R3.1/_tools_accuracy_checker_accuracy_checker_adapters_README.html
[convert_annotation]: http://docs.openvinotoolkit.org/2019_R3.1/_tools_accuracy_checker_accuracy_checker_annotation_converters_README.html
