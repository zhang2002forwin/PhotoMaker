FROM python:3.8

WORKDIR /app

COPY . . 

RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/  
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 

CMD [ "bash", "train.sh" ]