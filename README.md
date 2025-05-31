# ML System Design Document
## Система анализа и оценки информации о поведении пользователей для повышения эффективности работы умного браслета

Распределение ролей:
- Беломытцев Андрей – product owner, data science
- Фадеев Дмитрий – system architect

### 1. Цели и Предпосылки
#### 1.1. Зачем Идём в разработку продукта
##### 1.1.1. Бизнес-цель:
Улучшения эффективности обучения и продуктивности пользователей с помощью применения умного браслета, способного анализировать поведение пользователей и на основе этого давать обратную связь.
##### 1.1.2. Почему станет лучше, чем сейчас, от использования ML
Технология умного браслета нацелена на контролирование привычек пользователей с целью повешения их эффективности. Однако текущие технологии обработки данных не позволяют на достаточном уровне получать и оценивать информацию о поведении пользователя. Алгоритмы машинного обучения позволят значительно увеличить информацию, которая может быть задействована умным браслетом.

[BPMN диаграммы до и после внедрения системы](diagrams/business.md)

##### 1.1.3. Что будем считать успехом итерации с точки зрения бизнеса:
Успехом будет считаться разработка и интеграция технологий машинного обучения в умный браслет, который будет признан полезным для существенной группы пользователей.
#### 1.2. Бизнес-требования и ограничения
##### 1.2.1. Бизнес требования
1. Распознавание лиц для контроля просмотра контента
2. Распознавание настоящей и симулированной активности пользователя
3. Определение полезного и вредного контента по названию
4. Распознавание полезной деятельности по скриншотам
5. ИИ как учитель с наказанием
6. Поиск ключевых слов в названиях видео
##### 1.2.2. Бизнес ограничения
1. Ограничения на использование API для LLM
2. Стоимость: необходимо учитывать затраты на API при масштабировании
3. Пользовательский опыт: система не должна быть чрезмерно навязчивой или вызывать негативную реакцию у всех пользователей. Наказания должны быть настраиваемыми и применяться с осторожностью
4. Приватность: обработка скриншотов и данных с веб-камеры требует обеспечения конфиденциальности
#### 1.3. Функциональные требования
- Система должна получать видеопоток с веб-камеры пользователя.
- Система должна обнаруживать наличие человеческого лица в кадре видеопотока.
- Система должна отслеживать координаты курсора мыши пользователя.
- Система должна иметь возможность получать URL и заголовок (title) текущей активной веб-страницы.
- Система должна иметь возможность делать скриншоты текущего экрана пользователя.
- Система должна предоставлять интерфейс для взаимодействия пользователя с ИИ-учителем (например, на базе ChatGPT API).
- Система должна позволять загружать список "одобренных" YouTube-каналов.
- Система должна быть реализована как приложение для компьютера.
#### 1.4. Нефункциональные требования
- Latency: Модуль распознавания лиц должен обрабатывать видеопоток с задержкой, не мешающей восприятию пользователя (например, реакция в течение 1-2 секунд).
- Доступность: Система должна предоставлять пользователю понятную обратную связь о своем состоянии и обнаруженных событиях (например, причина применения стимула браслетом).
- Корректность: Система должна корректно обрабатывать ошибки API (например, проблемы с сетью, исчерпание лимитов, ошибки сервера API), предпринимать попытки повторных запросов (где это уместно) или информировать пользователя.
- Безопасность: API-ключи (ChatGPT, YouTube, Pavlok) должны храниться безопасным образом, предпочтительно не в открытом виде в коде клиентского приложения, если оно будет распространяться.
- Приватность: Пользователи должны быть явно проинформированы о том, какие данные собираются (видео с веб-камеры, движения мыши, скриншоты, история браузера), как они обрабатываются и передаются ли третьим сторонам (API).
#### 1.5. Процесс пилота и критерии успеха
N браслетов будут переданы пилотной группе из K человек, после чего вся информация о работе браслета будет сохраняться, а пилотная группа будет регулярно замерять изменения их состояния.
##### 1.5.2. Критерии успеха пилота
- Не менее X% пользователей пилотной группы сообщают об улучшении концентрации.
- Не менее Y% пользователей отмечают полезность функции ИИ-учителя.
- Техническая стабильность работы всех модулей на протяжении Z часов использования.
- Корректное срабатывание браслета в >90% случаев согласно логике программы.
#### 1.6. MVP и Технический долг
##### 1.6.1. MVP
- Реализация распознавания лиц для контроля просмотра контента.
- Реализация распознавания настоящей и симулированной активности пользователя.
- Реализация определения полезного и вредного контента по названию (преимущественно для видео).
- Реализация распознавания полезной деятельности по скриншотам с использованием внешних API (ChatGPT).
- Реализация прототипа ИИ-учителя с физическим подкреплением через браслет Pavlok.
- Разработка системы поиска ключевых слов в названиях видео для оценки их релевантности.
##### 1.6.2. Технический долг
- Создание собственных сложных нейросетевых моделей для анализа скриншотов "с нуля" (из-за отсутствия размеченных данных).
- Полноценный поиск по интернету для API ChatGPT в бесплатной версии.
- Поддержка широкого спектра программ и сайтов для распознавания полезности контента (фокус на универсальных признаках и видео).
- Детальная настройка интенсивности наказания в ИИ-учителе (пока только boolean).
- Разработка пользовательского интерфейса с высоким уровнем UX/UI (акцент на функциональности).
- Масштабное тестирование на большой выборке пользователей.
### 2. Методология
#### 2.1. Постановка задачи
**Что мы делаем с технической точки зрения**:
- **Распознавание лиц**: Задача бинарной классификации (лицо/глаза есть/нет) с использованием готовых моделей компьютерного зрения (например, cv2).
- **Распознавание активности**: Задача поиска аномалий / классификации на основе временных рядов (координаты мыши). Используется статистический показатель (энтропия) для разделения классов.
- **Определение полезного/вредного контента по названию**: Задача классификации видео с использованием LLM (ChatGPT API).
- **Распознавание полезной деятельности по скриншотам**: Задача мультимодальной классификации (image-text-to-text) с использованием LLM (ChatGPT API с Vision).
- **ИИ как учитель**: Задача генерации текста (обучающий материал, вопросы) и классификации ответов пользователя с использованием LLM (ChatGPT API со Structured Outputs). Управление внешним устройством (браслет) на основе классификации.
- **Поиск ключевых слов**: Задача извлечения информации (Information Extraction) и частотного анализа текста для построения списка релевантных терминов.
#### 2.2. Какие данные необходимы
- **Распознавание лиц**: Трекинг лица пользователя с веб-камеры при процессе чтения и просмотра контента
- **Распознавание активности**: данные о координатах мышки
- **Определение полезного/вредного контента по названию**: url видео
- **Распознавание полезной деятельности по скриншотам**: скриншоты пользователя
- **Поиск ключевых слов**: url видео
#### 2.3. Какие метрики качества будут использованы
- Accuracy: >85%
- F1-Score: >80%
#### 2.4. Риски на этапе анализа и планирования
- **Изменение условий/стоимости API**: Внешние API могут изменить цены, лимиты или прекратить поддержку.
- **Точность ML моделей**: Модели могут давать неточные или неадекватные результаты.
- **Этические соображения**: Связанные с приватностью, автономией пользователя, потенциальным злоупотреблением системой.
- **"Обман" системы**: Пользователи могут находить способы обходить ограничения.
### 3. Подготовка пилота
**Дизайн пилота**: исследование с контрольной группой. Участники пилотной группы будут использовать программу с браслетом в течение определенённого периода (1-2 недели) в повседневной жизни и при выполнении учебных/рабочих задач.
#### 3.1. Способ оценки пилота
1. **Субъективная оценка**: Анкетирование и интервью с пользователями для сбора обратной связи о полезности, удобстве использования, адекватности реакций системы и браслета. Оценка влияния на концентрацию, продуктивность, мотивацию.
2. **Объективные метрики (если возможно собирать)**: Время, проведенное за полезным контентом (по логам программы), количество выполненных учебных заданий (если применимо), частота срабатывания "наказаний" и "поощрений".
3. **Техническая оценка**: Сбор логов работы программы для анализа стабильности, скорости отклика, количества ошибок, частоты обращений к API.
#### 3.2. Что считаем успешным пилотом
1. **Пользовательская удовлетворённость**: Средняя оценка полезности системы по шкале от 1 до 5 не ниже 3.5.
2. **Улучшение продуктивности**: Не менее 60% пользователей сообщают о положительном влиянии системы на их концентрацию или продуктивность.
3. **Техническая стабильность**: Менее 5 критических сбоев на пользователя за время пилота.
4. **Эффективность функций**:
	- Распознавание лиц: >90% времени корректно определяет присутствие/отсутствие пользователя (по субъективной оценке пользователя).
	- Распознавание активности: >80% случаев симуляции корректно детектируются (по постановочным тестам пользователя).
	- Определение полезности контента: >75% совпадений с оценкой пользователя.
#### 3.3. Подготовка пилота
**Этапы подготовки**:
1. Отбор контрольной группы
2. Настройка браслета и инфраструктуры
3. Создание процедур мониторинга данных для оценок (субъективных и объективных)
4. Создание процедур эскалации проблем
5. Проведение исследования
### 4. Внедрение
#### 4.1. Архитектура решения

[Диагрмаа архитектуры](diagrams/architecture.md)

##### 4.1.2. Клиентское приложение
- **Модуль сбора данных**: Захват видео с веб-камеры (OpenCV), отслеживание мыши (pyautogui/pynput), получение URL и заголовков (через API браузера или библиотеки), создание скриншотов (pyautogui).
- **Модули анализа**:
- Локальный анализ: Распознавание лиц (OpenCV), анализ энтропии мыши.
- Удаленный анализ: Клиенты для ChatGPT API (анализ названий, скриншотов, ИИ-учитель), Pavlok API (управление браслетом), YouTube API (сбор названий видео).
- **Модуль принятия решений**: Логика, агрегирующая результаты анализа и определяющая действия.
- **Модуль взаимодействия с пользователем**: веб-UI
##### 4.1.3. Внешние сервисы (API):
- **OpenAI API**:
	- Методы: `chat/completions` (для текстовых задач и Vision).
- **Pavlok API**:
	- Методы: `stimulus/send` (для отправки стимулов).
- **YouTube Data API v3**:
	- Методы: `playlistItems/list` (для получения названий видео).
#### 4.2. Описание инфраструктуры и масштабируемости
-   **Клиентская часть**: Запускается локально на ПК пользователя. *Причина*: Прямой доступ к периферии (веб-камера, мышь, экран), снижение задержек для локального анализа.
-   **Серверная часть (внешние API)**: Используются облачные сервисы OpenAI, Pavlok, Google. *Причина*: Нет необходимости разворачивать и поддерживать собственные сложные ML-модели и инфраструктуру для них.

-   **Плюсы**: Быстрая разработка прототипа, использование state-of-the-art моделей без затрат на их обучение, отсутствие необходимости в собственной серверной инфраструктуре для ML.
-   **Минусы**: Зависимость от сторонних API (доступность, стоимость, изменения условий), ограничения бесплатных тарифов, потенциальные проблемы с задержками сети, вопросы конфиденциальности данных при отправке на внешние серверы.

[Диаграмма структуры данных](diagrams/data.md)

#### 4.3. Требования к работе системы
- **SLA**: 99.95% доступности
- **Latency**:
	- API запросы: <200ms
	- Web-интерфейс: <1s на загрузку
#### 4.4. Риски
-   **Изменение условий/стоимости API**: Внешние API могут изменить цены, лимиты или прекратить поддержку. *Митигация: Предусмотреть возможность замены API, отслеживать изменения.*
-   **Точность ML моделей**: Модели могут давать неточные или неадекватные результаты. *Митигация: Тщательное тестирование, возможность пользовательской калибровки, сбор обратной связи для улучшения промптов/моделей.*
-   **Пользовательское принятие**: Не все пользователи положительно воспримут идею "наказаний" или постоянного мониторинга. *Митигация: Гибкие настройки, возможность отключения отдельных функций, фокус на позитивном подкреплении.*
-   **Технические проблемы**: Сбои в работе ПО, проблемы с интернет-соединением, несовместимость с ОС пользователя. *Митигация: Тщательное тестирование, логирование, предоставление поддержки.*
-   **Этические соображения**: Связанные с приватностью, автономией пользователя, потенциальным злоупотреблением системой. *Митигация: Прозрачность работы системы, пользовательский контроль, соблюдение этических принципов разработки ИИ.*
-   **"Обман" системы**: Пользователи могут находить способы обходить ограничения. *Митигация: Постоянное улучшение алгоритмов детекции, но признание, что 100% защита невозмож


[Диаграмма Последовательности Пользования системой](diagrams/sequence.md)



## Реализация

Задача: Применение анализа данных, нейронных сетей, машинного обучения и т.д. для улучшения эффективности работы браслета, обучения и повышения продуктивности.

Направления работы:
- Распознавание лиц для контроля просмотра контента
- Распознавание настоящей и симулированной активности пользователя
- Определение полезного и вредного контента по названию
- Распознавание полезной деятельности по скриншотам
- ИИ как учитель с наказанием
- Поиск ключевых слов в названиях видео

Результат будет считаться успешным, если полученные технологии можно будет подключить к браслету и они будут полезны некоторым пользователям, не обязательно всем.

### Распознавание лиц для контроля просмотра контента

В программу для компьютера добавлено распознавание лиц с веб-камеры, чтобы контролировать чтение и просмотр контента. Если камера видит лицо и/или глаза, то это значит, что пользователь смотрит в экран и читает либо смотрит видео, а не ушёл в другое место и смотрит тик-токи на телефоне, пока на экране ноутбука выведена полезная информация. В дополнение к другим частям программы, которые проверяют открыт ли правильный контент на весь экран.

Для распознавания лица используется библиотека cv2.

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = 0
t = 0
def face():
  global cap
  global t
  t = 0
  try:
    if cap == 0:
      cap = cv2.VideoCapture(0)
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 19)
    eyes = eye_cascade.detectMultiScale(img_gray, 1.1, 19)
    return faces != () or eyes != ()
  except:
    print('FACE ERROR')
    return False
```

### Распознавание настоящей и симулированной активности пользователя

Для обмана системы проверки активности пользователь может просто вращать мышкой одной рукой, а второй рукой пользоваться телефоном, а программа будет засчитывать это как полезную активность. Вращение позволяет постоянно менять координату мышки, что простейшим алгоритмом считается как постоянное выполнение полезной задачи.

Для исправления этой проблемы можно проверять энтропию движений. В результате экспериментов выяснилось, что простое вращение мышки имитирующее активность имеет энтропию ~5.4, в то время как реальная деятельность имеет энтропию ~6.4. Что позволяет провести границу между реальными и искусственными движениями, например, на значении 6.0, реальные < 6.0, искусственные > 6.0. Это происходит в результате того, что реальные движения более резкие, чем искусственные. У реальных движений есть цель быстрее добраться до нужной кнопки и нажать по ней. Такие движения тоже можно эмитировать, но это уже намного менее удобно, чем простое вращение мыши. Мышь нужно резко двигать в разные стороны на разное расстояние, что требует больших усилий.

Для получения данных о координатах мыши используется pynput, а для расчётов numpy и scipy.stats.entropy.

```python
import numpy as np
from pynput import mouse
from scipy.stats import entropy

mouse_positions = [(0, 0)] * 200

def on_move(x, y):
  mouse_positions.append((x, y))
  positions_array = np.array(mouse_positions[-100:])
  distances = np.sqrt(np.sum(np.diff(positions_array, axis=0)**2, axis=1))
  print(entropy(distances, base=2))

with mouse.Listener(on_move=on_move) as listener:
  try:
    listener.join()
  except KeyboardInterrupt:
    listener.stop()
```

### Определение полезного и вредного контента по названию

Использование API ChatGPT для определения полезный ли контент на текущем сайте или нет только по URL не будет работать, т.к. в бесплатной версии API нет поиска в интернете. Поэтому в этом случае нужно предоставлять дополнительную информацию в запросе, которую наша программа должна собирать самостоятельно. Отправка полного HTML кода страницы не даёт хороших результатов, т.к. в ней слишком много лишней информации не дающей понять суть страницы и требующей много времени на отправку. Придётся ограничиваться заголовком страницы или текстом извлеченным из HTML. Плюс пользователь может достаточно быстро перемещаться между страницами, что в условиях ограниченного количества отправляемых запросов к API может испортить опыт.

Такую систему значительно удобнее использовать с видео, т.к. можно отправить заголовок видео и другую информацию и по ним определить содержание видео. Плюс переключение между видео происходит медленнее, чем между страницами сайтов, что позволяет не перегружать API.

Проверка нескольких видео показала, что описание тоже бывает полезным в определении полезности видео. Например, полезные видео отмеченные как false без описания, после добавления описания отмечались как true.

```python
import json
import requests
from openai import OpenAI

client = OpenAI()

YT_API_KEY = '...'

url = 'https://www.youtube.com/watch?v=...'

video = requests.get(f'https://www.googleapis.com/youtube/v3/videos?id={url[32:43]}&part=snippet&key={YT_API_KEY}').json()
video = video['items'][0]['snippet']

title = video['title']
channel = video['channelTitle']
description = video['description']

print('title:', title)
print('channel:', channel)
print('description:', repr(description))

response = client.chat.completions.create(
  model='gpt-4o-mini',
  messages=[
    {
      'role': 'user',
      'content': [
        {
          'type': 'text',
          'text': f'This is YouTube video information. Is this video related to programming?\nTitle: {title}\nChannel: {channel}\nDescription: {description}'
        }
      ],
    }
  ],
  response_format={
    'type': 'json_schema',
    'json_schema': {
      'name': 'conditioning',
      'schema': {
        'type': 'object',
        'properties': {
          'reinforcement': {
            'type': 'boolean',
            'description': 'True if the URL related to programming. Else False.'
          },
          'text': {
            'type': 'string',
            'description': 'Main text response.'
          }
        },
        'required': ['reinforcement', 'text'],
        'additionalProperties': False
      },
      'strict': True
    }
  },
  max_tokens=500
)

result = json.loads(response.choices[0].message.content)

print('reinforcement:', result['reinforcement'])
print('text:', result['text'])
```

### Распознавание полезной деятельности по скриншотам

Слишком сложно было бы обучить нейросеть самостоятельно понимать по скриншотам, полезный ли контент на нём представлен или нет, как минимум потому что нет достаточного количества размеченных скриншотов во множестве различных программ и сайтов. Также очень важно распознавание текста и понимания нейросетью, что этот текст значит.

Поэтому для данной задачи подойдут уже обученные нейросети типа [image-text-to-text по классификации huggingface](https://huggingface.co/models?pipeline_tag=image-text-to-text). Вроде [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) или ChatGPT. Проверил Llama на нескольких скриншотах, правильно определяет, но с использованием её есть некоторые сложности. Поэтому в данном проекте будет применять ChatGPT API.

Для работы с такими нейросетями мог бы использоваться prompt вроде такого:

> This is a screenshot. Is the user learning programming? If yes answer must contain word "REINFORCEMENT", if no "PUNISHMENT". Answer must contain "REINFORCEMENT" or "PUNISHMENT"!

А затем в полученном в ответ сообщении было искались бы слова "REINFORCEMENT" или "PUNISHMENT".

Но данный способ не надёжен, т.к. нейросети иногда забывают написать эти слова, благо у ChatGPT есть [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), что позволяет заставить нейросеть ответить в определённом формате, в т.ч. в виде Boolean значения.

У API ChatGPT есть [ограничения](https://platform.openai.com/docs/guides/rate-limits), которые нужно учитывать при его использовании. Для бесплатной версии разрешено не более 3 запросов в минуту к модели gpt-4o-mini. Поэтому нам придётся ограничить частоту отправки скриншотов.

Проверка скриншотов хорошо сочетается с проверкой активности пользователя. Деятельность будет считаться правильной только при одобрении скриншота нейросетью и при наличии активности пользователя.

Скриншот делается с помощью pyautogui, преобразовывается в нужный формат с помощью BytesIO из io и base64, затем отправляется на API ChatGPT с помощью библиотеки openai.

В результате мы получаем boolean значение и текст с пояснением соответствующего решения.

```python
import json
import base64
import pyautogui
from io import BytesIO
from openai import OpenAI

client = OpenAI()

# def encode_image(image_path):
#   with open(image_path, 'rb') as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')

# image_path = '1.png'
# base64_image = encode_image(image_path)

screenshot = pyautogui.screenshot()
buffer = BytesIO()
screenshot.save(buffer, format='PNG')
buffer.seek(0)
base64_image = base64.b64encode(buffer.read()).decode('utf-8')
buffer.close()

response = client.chat.completions.create(
  model='gpt-4o-mini',
  messages=[
    {
      'role': 'user',
      'content': [
        {
          'type': 'text',
          'text': 'This is a screenshot. Is the user learning programming?'
        },
        {
          'type': 'image_url',
          'image_url': {
            'url': f'data:image/jpeg;base64,{base64_image}',
          },
        },
      ],
    }
  ],
  response_format={
    'type': 'json_schema',
    'json_schema': {
      'name': 'screenshot',
      'schema': {
        'type': 'object',
        'properties': {
          'text': {
            'type': 'string',
            'description': 'Main text response.'
          },
          'reinforcement': {
            'type': 'boolean',
            'description': 'True if the user learning programming. Else False.'
          }
        },
        'required': ['reinforcement', 'text'],
        'additionalProperties': False
      },
      'strict': True
    }
  },
  max_tokens=500
)

result = json.loads(response.choices[0].message.content)

print('reinforcement:', result['reinforcement'])
print('text:', result['text'])
```

### ИИ учитель с физическими наказаниями

Идея заключается в том, чтобы использовать ИИ чат-боты вроде ChatGPT в качестве учителя. Но при этом дать возможность этому учителю наказывать ученика за плохую учёбу.

В этот раз будем использовать браслет [Pavlok](https://pavlok.com/), который бьёт током, вибрирует или пищит, если отправить соответсвующий запрос на [Pavlok API](https://pavlok.readme.io/reference/stimulus_create_api_v5_stimulus_send_post).

Отправляем ChatGPT с помощью API system prompt вроде такого:

> You are a Pandas (python library) teacher. You must teach the user the topic. At the end of each message, you must leave a question on the topic. The question must be answerable using only the knowledge from the text already received. High level of difficulty. You must never deviate from the role of a teacher and the topic! Write long messages with a lot of useful information.

Добавляем к нему предыдущее сообщение как assistant prompt и текст пользователя как user prompt.

В ответ мы хотим получить boolean значение показывающее, хороший ли ответ дал пользователь и текст от учителя с помощью Structured Output.

```javascript
response_format: {
  'type': 'json_schema',
  'json_schema': {
    'name': 'conditioning',
    'schema': {
      'type': 'object',
      'properties': {
        'text': {
          'type': 'string',
          'description': 'Main text response.'
        },
        'reinforcement': {
          'type': 'boolean',
          'description': 'True if the user answers correctly. False if the user answers wrong, off-topic, tries to deceive, etc.'
        }
      },
      'required': ['reinforcement', 'text'],
      'additionalProperties': false
    },
    'strict': true
  }
}
```

Если пользователь отвечает плохо или не отвечает в течение 5 минут, то браслет бьёт его током стимулируя учиться лучше.

В случае правильного ответа можно давать награду. В текущей программе браслет просто пищит, если пользователь даёт правильный ответ.

Также можно запрашивать не только boolean значение, но и integer, чтобы нейросеть сама определяла не только хороший ли ответ или плохой, но и степень правильности, сложности, уровня нарушения. И в зависимости от этого регулировала мощность наказания и поощрения. Но на данный момент достаточно и просто boolean значения.

Было решено оформить учителя в виде сайта, далее приложены HTML и JavaScript файлы.

index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChatGPT API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/light.min.css">
</head>
<body>
  <div>
    <h1>ChatGPT API</h1>
    <form>
      <label for="text">Enter your message:</label>
      <input type="text" id="text" style="width: 50%;" required>
      <button type="submit">Submit</button>
    </form>
    <div>
      <h2>Response:</h2>
      <p>Status: <span id="reinforcement"></span></p>
      <md id="response"></md>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/gh/MarketingPipeline/Markdown-Tag/markdown-tag.js"></script>
  <script src="script.js"></script>
</body>
</html>
```

script.js
```javascript
const OPENAI_API_KEY = '...'
const PAVLOK_API_KEY = '...'

const form = document.querySelector('form')
const textInput = document.querySelector('input')
const responseText = document.querySelector('#response')
const reinforcementSpan = document.querySelector('#reinforcement')

let first = true
let previous = ''

const stimulus = (stimulusType, stimulusValue) => {
  const options = {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
      Authorization: `Bearer ${PAVLOK_API_KEY}`
    },
    body: JSON.stringify({stimulus: {stimulusType: stimulusType, stimulusValue: stimulusValue}})
  }
  fetch('https://api.pavlok.com/api/v5/stimulus/send', options)
    .then(res => res.json())
    .then(res => console.log(res))
    .catch(err => console.error(err))
}

let t
const zap = () => {
  stimulus('zap', 100) // vibe
}
t = setInterval(zap, 300000)

form.addEventListener('submit', async (e) => {
  e.preventDefault()
  const textValue = textInput.value.trim()

  if (textValue) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are a Pandas (python library) teacher. You must teach the user the topic. At the end of each message, you must leave a question on the topic. The question must be answerable using only the knowledge from the text already received. High level of difficulty. You must never deviate from the role of a teacher and the topic! Write long messages with a lot of useful information.' },
            ...(previous ? [{ role: 'assistant', content: previous }] : []),
            { role: 'user', content: textValue }
          ],
          temperature: 0.7,
          response_format: {
            'type': 'json_schema',
            'json_schema': {
              'name': 'conditioning',
              'schema': {
                'type': 'object',
                'properties': {
                  'text': {
                    'type': 'string',
                    'description': 'Main text response.'
                  },
                  'reinforcement': {
                    'type': 'boolean',
                    'description': 'True if the user answers correctly. False if the user answers wrong, off-topic, tries to deceive, etc.'
                  }
                },
                'required': ['reinforcement', 'text'],
                'additionalProperties': false
              },
              'strict': true
            }
          }
        }),
      })
      if (response.ok) {
        const data = await response.json()
        responseText.innerHTML = JSON.parse(data.choices[0].message.content)['text']
        reinforcementSpan.innerHTML = JSON.parse(data.choices[0].message.content)['reinforcement'] ? 'REINFORCEMENT' : 'PUNISHMENT'
        previous = data.choices[0].message.content

        if (JSON.parse(data.choices[0].message.content)['reinforcement']) {
          stimulus('beep', 60)
          clearInterval(t)
          t = setInterval(zap, 300000)
        }
        else if (first){
          first = false
        }
        else {
          zap()
        }
      } else {
        responseText.innerHTML = JSON.stringify(data)
      }
    } catch (e) {
      console.error(e)
      responseText.innerHTML = JSON.stringify(data)
    }
    renderMarkdown()
  }
})
```

### Поиск ключевых слов в названиях видео

Один из самых простых способов определения правильного видео это проверка его на наличие специальных слов вроде "python", "javascript" и т.д. Полезно было бы составить большой список таких слов. Хотя это можно делать вручную, я попробую обнаружить такие слова с помощью анализа заголовков каналов, которые уже размечены мною как правильные. Такой список можно загрузить в расширение и нагрузка при определении правильных видео будет минимальная, в отличие от более сложных способов анализа вроде нейросетей. А высокая эффективность нужна, т.к. правильные слова нужно искать не только в видео, которое пользователь смотрит прямо сейчас, но и во всех отображаемых списках видео, чтобы подсветить хорошие и спрятать плохие видео.

Так же пробовал использовать ChatGPT для генерации списка таких слов, но он способен выдать не более 100 слов, дальше идут длинные словосочетания, которые использовать бесполезно.

Prompt: Напиши 100 слов через запятую, по наличию которых можно определить связан ли заголовок с изучением программирования или нет.
Примеры: python, javascript, html, matplotlib, django, ...

Ответ:

> java, ruby, kotlin, swift, typescript, react, angular, vuejs, laravel, flask, express, nodejs, spring, hibernate, c++, c#, go, rust, scala, perl, bash, sql, nosql, mongodb, postgresql, mysql, oracle, redis, sqlite, jupyter, anaconda, pandas, numpy, scipy, seaborn, plotly, keras, opencv, beautifulsoup, scrapy, selenium-webdriver, puppeteer, jenkins, travis, circleci, gitlab, bitbucket, codewars, leetcode, hackerrank, algorithm, data structure, recursion, sorting, searching, binary tree, linked list, stack, queue, graph, dynamic programming, big o, complexity, software development, agile, scrum, kanban, devops, microservices, api, rest, graphql, web scraping, machine learning, deep learning, artificial intelligence, neural networks, reinforcement learning, cloud computing, aws, azure, google cloud, virtualization, docker-compose, kubernetes, containerization, api gateway, load balancer, firewall, security, encryption, authentication, authorization, oauth, jwt, web development, mobile development, game development, desktop applications, user interface

Получение заголовков с помощью API:

```python
import requests
import json
```

```python
x = requests.get('https://.../.json').json()
x = [i['id'] for i in x]
with open('good-channels.txt', 'w') as f:
  print(*x, sep='\n', file=f)
```

```python
YT_API_KEY = '...'

with open('good-channels.txt') as f:
  x = [i.strip() for i in f.readlines()]

r = []
page_count = 4

for channel_id in x:
  maxResults = 50
  uploads = 'UULF' + channel_id[2:]
  nextPageToken = ''
  for _ in range(page_count):
    videos = requests.get(
      f'https://www.googleapis.com/youtube/v3/playlistItems?part=snippet%2CcontentDetails&maxResults={maxResults}&playlistId={uploads}&key={YT_API_KEY}{"&pageToken=" + nextPageToken if nextPageToken else ""}'
    ).json()
    try:
      for m in videos['items']:
        try:
          m = m['snippet']
          if m['resourceId']['kind'] == 'youtube#video':
            r += [m['title']]
        except Exception as e:
          print(e)
      if 'nextPageToken' in videos:
        nextPageToken = videos['nextPageToken']
      else:
        break
    except Exception as e:
      print(e)
      break
with open('good-titles.txt', 'w') as f:
  print(*r, sep='\n', file=f)
```

Анализ залоговков:

```python
from collections import Counter
import re
import wordfreq
import string
import pymorphy2
import nltk
from nltk.corpus import stopwords

with open('good-titles.txt') as f:
  text = f.read()
```

```python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

morph = pymorphy2.MorphAnalyzer()
words = nltk.word_tokenize(text.lower())
words = [
          morph.parse(i)[0].normal_form
          for i in words if
          i not in stopwords.words('english') and
          i not in stopwords.words('russian') and
          i not in string.punctuation
        ]
```

```python
words = re.findall(r'\b\w+\b', text.lower())
```

```python
words = re.findall(r'\d+', text.lower())
```

```python
word_counts = Counter(words)
```

Деление на частоту слов, полученную из модуля wordfreq позволит избавиться от слов часто встречаемых и в обычном тексте.

```python
word_counts = {
  word: int(count / (max(wordfreq.word_frequency(word, 'ru', wordlist='best'), wordfreq.word_frequency(word, 'en', wordlist='best')) + 0.0001))
  for word, count in word_counts.items()
}
```

```python
frequent_words = sorted(list(word_counts.items()), key=lambda i: i[1])
frequent_words = frequent_words[-100:]
```

```python
for word, count in frequent_words:
  # print(word, end=', ')
  print(f'{word}: {count}')
```

Топ 100 слов без деления:

> и, в, по, на, in, to, егэ, the, как, tutorial, how, с, a, with, and, для, for, 2, python, of, 1, что, 3, огэ, умскул, 5, 10, css, за, урок, из, не, is, 2024, i, к, 2023, js, s, о, photoshop, 4, история, adobe, react, on, what, javascript, design, английский, от, using, create, часть, разбор, html, beginners, математика, why, все, 8, resolve, you, задание, 7, почему, c, 6, data, your, davinci, course, класс, blender, part, нуля, 9, code, физике, vs, 2025, flutter, ai, курс, pro, 11, illustrator, from, минут, язык, full, new, make, science, java, лекция, до, такое, it, arduino

Результаты без деления на частоту слов, есть много лишних часто употребимых слов, но всё равно можно вычленить много полезных слов.

Топ 100 слов с делением:

> егэ, tutorial, python, огэ, умскул, css, js, урок, photoshop, adobe, react, javascript, разбор, beginners, davinci, 2024, html, математика, blender, resolve, 2023, flutter, английский, нуля, физике, illustrator, задание, ai, arduino, лекция, java, create, класс, esp32, premiere, задания, php, физика, математике, 2025, vs, биология, design, начинающих, история, 10, информатике, химия, pro, курс, уроки, django, api, обзор, программирование, code, биологии, английском, 3d, explained, animated, node, топ, effects, responsive, язык, 2022, обществознанию, animation, химии, cc, effect, c, science, web, data, обществознание, using, фипи, видеоурок, spring, английскому, информатика, build, задача, академия, boot, языку, задачи, app, tips, 100, минут, 2021, трушин, documentary, learn, course, text, esp8266

Результаты при делении на частоту слов уже намного чище и из них уже намного легче находить нужные и добавлять в список.

Топ 100 слов с примененим nltk, pymorphy2 с делением:

> егэ, tutorial, python, '', умскула, огэ, css, –, урок, английский, ``, задание, физика, —, adobe, photoshop, «, », javascript, react, математик, разбор, beginners, класс, davinci, биология, html, blender, нуль, resolve, химия, 2024, flutter, лекция, минута, //, информатика, 2023, illustrator, язык, задача, ai, история, arduino, java, обществознание, курс, программирование, математика, create, ’, premiere, js, начинающий, design, php, подготовка, esp32, 2025, vs, вариант, pro, ..., django, миф, 10, api, функция, code, обзор, вебиум, explained, animated, география, effects, приложение, responsive, самый, 3d, ||, решать, русский, ошибка, animation, теория, cc, effect, next.js, основа, science, n't, 2022, using, data, web, фипеть, балл, видеоурок, часть, spring

Топ 100 слов с примененим nltk, pymorphy2 без деления:

> егэ, tutorial, python, английский, урок, '', умскула, огэ, история, задание, css, –, 2, 10, 1, класс, физика, 5, 3, язык, ``, 2024, photoshop, adobe, —, часть, «, react, », 2023, javascript, design, using, курс, create, задача, 's, математик, разбор, beginners, resolve, html, биология, 4, химия, почему, davinci, самый, data, course, blender, нуль, part, минута, code, flutter, лекция, такой, pro, ai, год, информатика, //, русский, 2025, vs, новый, 7, illustrator, вариант, new, make, full, science, java, 8, arduino, 6, подготовка, математика, effect, effects, обществознание, learn, который, это, premiere, программирование, 9, english, build, ’, js, начинающий, 11, spring, explained, 2022, text, php

Видно, что применение nltk и pymorphy2 пользы особой не принесло. Обработка данных с ними заняла намного больше времени. Выяснилось, что не достаточно использовать i not in string.punctuation для удаления ненужных символов, т.к. в названиях видео часто используются более сложные символы и сочетания, чем в обычном тексте. pymorphy2 ещё неправильно привёл слова к начальной форме "умскул", где "скул" означает school, pymorphy2 воспринял, как слово "скула", поэтому привёл к форме "умскула". Такие изменения форм слова очень вредны для определения ключевых слов, т.к. после их нахождения, будет проверяться точное совпадение в расширении.

Ради интереса `r'\b\w+\b'` было заменено на `r'\d+'`, чтобы посмотреть есть ли числа, а не слова, которые могут быть использованы в качестве индикатора полезного контента.

В результате чего получилось две основные группы полезных чисел:
1. Даты
  - 1812, 1905, 1914, 1917, 1941, 1945, 2008
2. Названия различных технологий, например электроника
  - 18650 (популярный формат аккумуляторов)
  - 8266 (часть названия Wi-Fi модуля ESP8266)
  - 4056 (часть названия модуля зарядки аккумуляторов TP4056)

И две основные группы бесполезных чисел:
1. Даты относящиеся к настоящему времени
  - 2020, 2021, 2022, 2023, 2024, 2025
2. Номера видео или номера заданий в ОГЭ/ЕГЭ

В результате было найдено очень много полезных слов, часть которых я добавил и в свой список внутри расширения.
