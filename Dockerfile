FROM ubuntu:18.04

RUN apt-get update
RUN apt-get -y install apt-transport-https ca-certificates gnupg software-properties-common wget 
RUN apt-get -y install python3.7 python3-pip
RUN apt-get -y install git

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update
RUN apt-get -y install cmake

RUN apt-get -y install freeglut3-dev
RUN apt-get -y install libgl1-mesa-dev
RUN apt-get -y install libgtest-dev

WORKDIR /GradWork

COPY . /GradWork

RUN pip3 install -r requirements.txt

CMD ["./run.sh"]
