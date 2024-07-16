FROM tensorflow/tensorflow

RUN apt-get -qq update \
    && apt-get -qq -y install git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean \
    # && apt-get install vim \
    # && apt-get install libgomp1

RUN mkdir app

COPY ./ ./app

WORKDIR /app

RUN pip install --upgrade pip \
    pip install -r requirements.txt \
    poetry install \
    poetry update 

