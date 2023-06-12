# Atrial fibrillation termination

Predicting spontaneous termination of atrial fibrillation (AF) as an edge computing architecture. Network params can be found [here](/af-termination//neural-network/README.md)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

Hardware:

- Arduino nano 33 BLE Sense Lite
- ESP32-S3-DevKitM-1

Software:

- wfdb python package
- tensorflow
- [tflite for arduino](https://github.com/tensorflow/tflite-micro-arduino-examples)
- poetry

### Installing

Use poetry to install the virtual enviroment.

```bash
poetry install
poetry shell
```

## Built With

To build software on hardware I used Arduino IDE.

## Authors

- **Piotr Baryczkowski** - *All the work* - [Piotr45](https://github.com/Piotr45)

## Acknowledgments

- [AF Termination Challenge Database](https://physionet.org/content/aftdb/1.0.0/)

## TODO

- Update README
- Rewrite train notebook to python file
- Clean code
