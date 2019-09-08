# Base Images
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3
MAINTAINER eguil <liuge1229@foxmail.com>

ADD ./code /competition/code/
ADD ./model/chinese-bert_chinese_wwm_pytorch/ /competition/model/chinese-bert_chinese_wwm_pytorch/
ADD ./model/chinese_wwm_ext_pytorch/ /competition/model/chinese_wwm_ext_pytorch/
ADD ./model/ERNIE/ /competition/model/ERNIE/

ADD ./model/best_bert_model /competition/model/
ADD ./model/best_bert_model_0907 /competition/model/
ADD ./model/best_ext_model /competition/model/
ADD ./model/best_ernie_model /competition/model/

ADD ./Dockerfile /competition/Dockerfile
ADD ./requirements.txt /competition/requirements.txt
ADD ./run.sh /competition/run.sh

WORKDIR /competition

#RUN rm -rf /var/lib/apt/lists/*
#RUN apt-get clean
#RUN apt-get update
#RUN apt-get install gcc -y
RUN pip --no-cache-dir install  -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN chmod -R +w /competition
CMD ["sh", "run.sh"]