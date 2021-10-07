#From this homework, I learnt to separately tag and push a docker image accordingly - subramanivasu/zoomcamp_hw_model:3.8.12-slim . For the hw,the specified docker image has to be used.
FROM agrigorev/zoomcamp-model:3.8.12-slim

#Installing pipenv 
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]

#Installing pipenv dependencies
RUN pipenv install --system --deploy

COPY ["predict.py","model1.bin","dv.bin","./"]

#The port is running in the host machine and and we expose the 9696 port to establish the connection.
EXPOSE 9696 

#The Entrypoint says how the container will run. It will run the predict server with gunicorn.
ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9696","predict:app"]