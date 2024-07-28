FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /usr/src

COPY data/*.csv data/
COPY Model/ Model/
COPY *.py .
COPY requirements.txt .

RUN pip install --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "auto_nn.py"]