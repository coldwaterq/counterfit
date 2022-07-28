FROM nvcr.io/nvidia/cuda:11.7.0-base-ubuntu20.04
RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-pip
WORKDIR /workdir
RUN python3 -m pip install --upgrade pip
COPY requirements.txt /workdir/requirements.txt
RUN python3 -m pip install -r requirements.txt --ignore-installed
COPY counterfit /workdir/counterfit
COPY examples/scripting/miface.py /workdir/test/main.py

ENTRYPOINT ["python3"]
CMD ["test/main.py"]
