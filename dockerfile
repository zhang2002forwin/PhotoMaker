FROM python:3.8

WORKDIR /app

COPY . . 

RUN pip install pip3 install torch torchvision torchaudio 
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 

CMD [ "bash", "train.sh" ]