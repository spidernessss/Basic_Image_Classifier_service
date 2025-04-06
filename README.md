<!-- О Проекте -->
## О проекте
Базовый image classifier построенный на основе `MobileNet` по весам от `ResNet`. <br/>
Веб-интерфейс (JS, HTML, CSS), backend написан на `FastApi`. <br/>
Пользователь загружает картинку для классификации и датасет, после нажимает на кнопку classify и получает изображение с текстом определённого класса.
Сопоставление изображения с датасетом происходит посредством сравнения feature vectors в векторной бд `milvus`. <br/>
Сборка - `docker-compose up -d` по умолчанию запускается на localhost:8010

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Технологии

* [Python](https://www.python.org/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Milvus](https://milvus.io/ru)
* [Docker](https://www.docker.com/)
* [HuggingFace](https://huggingface.co/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Видео собранной и настроенной версии



https://github.com/user-attachments/assets/460c24a1-b99d-4d43-83ec-070e78036117



