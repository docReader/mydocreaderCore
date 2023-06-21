# FROM avi1504018/tf:cuda112
FROM python:3.8.10
LABEL mantainer="Sazzad Hossain"

RUN pip install --upgrade pip
RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config
RUN pip install pytesseract
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/ben.traineddata
RUN mv ben.traineddata /usr/share/tesseract-ocr/4.00/tessdata/ben.traineddata


WORKDIR /a2i-drv/src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "./src/api.py"]
