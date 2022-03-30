# Imagen tensorflow (https://github.com/ARM-software/Tool-Solutions/tree/master/docker/tensorflow-aarch64)
FROM armswdev/tensorflow-arm-neoverse:r21.12-tf-2.7.0-eigen

# Instalar dependencias
RUN sudo apt-get update 
RUN sudo apt-get install --no-install-recommends -y libmysqlclient-dev python3.8-venv mysql-server

# Preparación entorno virtual 
WORKDIR /home/ubuntu
#RUN python3 -m venv venv
ENV PATH="/home/ubuntu/python3-venv/bin:$PATH"
ENV VIRTUAL_ENV=/home/ubuntu/python3-venv

ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD app app
ADD webapp.py webapp.py
ADD config.py config.py
RUN pip install SQLAlchemy

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1
ENV CONFIG_ENVIRONMENT=development
ENV MYSQL_HOST=mysql_server
ENV MYSQL_USER=root
ENV MYSQL_PASSWORD=root
ENV MYSQL_DB=FaysPy
ENV MYSQL_PORT=3306
ENV PORT=5000
EXPOSE 5000
ENV TWITTER_CONSUMER_KEY=I4zDy3Y0BkJjPDLgfMEy0ytNS
ENV TWITTER_CONSUMER_SECRET=MQ9fCfeyUJnYVKAxRHABOQrqECjKOeHgrQfPB1pfCajd7pYd6u
ENV TWITTER_ACCESS_TOKEN=1495840711366918147-nTrnWgwIybwp7hfOH0I6X7C1j5o4BU
ENV TWITTER_ACCESS_TOKEN_SECRET=wmjfujIYTpJKQfHcg9XKsRauvpB8044wavENXzJJNvW5t
ENV MODEL_FILEPATH="/home/ubuntu/app/nlp/models/"
# Activar entorno virtual
ENV VIRTUAL_ENV=/home/ubuntu/code/venv
ENV PATH="/home/ubuntu/python3-venv/bin:$PATH"

CMD /usr/bin/env /home/ubuntu/python3-venv/bin/python /home/ubuntu/webapp.py 
#CMD tail -f /dev/null