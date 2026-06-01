FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    python3-devel \
    && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY app ${LAMBDA_TASK_ROOT}/app

CMD ["app.lambda_handler.handler"]