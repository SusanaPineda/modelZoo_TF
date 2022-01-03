# Script model zoo Tensorflow
## Instalacion
* instalar requirements.txt
* Descargar el repositorio tensorflow/models
````
git clone https://github.com/tensorflow/models.git
````
* instalar protobuf.
````
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

````
* Descarga de los [modelos](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


## Instalar Tensorflow en arquitecturas jetson
Antes de tratar de instalar modelZoo en las arquitecturas jetson,
se debe instalar tensorflow siguiendo las [instrucciones](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)proporcionadas
por el fabricante.

* Instalar el Jetpack en el dispositivo jetson o comprobar la versión instalada
````
sudo apt-cache show nvidia-jetpack
````
* Instalar dependencias
````
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools

sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
````

* Instalar tensorflow teniendo en cuenta la versión a utilizar y la versión instalada del jetpack. Se puede comprar en la [documentación](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel) proporcionada por el fabricante.
````
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow==$TF_VERSION+nv$NV_VERSION
````

## Instalar matplotlib en arquitecturas jetson
````
sudo apt-get install python-matplotlib
````
