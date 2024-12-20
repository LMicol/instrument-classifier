# Musical Instrument Sound Classifier

Welcome to the **Musical Instrument Sound Classifier** repository!

This project utilizes machine learning to classify musical instrument sounds using Mel Spectrogram features extracted from audio files.

The repository is structured to facilitate easy exploration, experimentation, and deployment of the classifier.

---

## Table of Contents
1. [About the Project](#about-the-project)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Documentation](#documentation)
5. [Contributing](#contributing)
6. [License](#license)

---

## About the Project
This project aims to classify sounds of musical instruments such as guitar, piano, drums, and violin. Key highlights include:
- Mel Spectrogram feature extraction for audio preprocessing.
- Pre-trained and custom-trained models for experimentation.
- Progressive improvements documented through various model iterations.
- Deployment-ready server for real-time classification.

### Features
- **Multiple Models:** Six different models, each with unique approaches, are trained and evaluated.
- **Visualization:** Confusion matrices and model performance metrics are documented.
- **Deployment:** Dockerized server and web interface for easy deployment.
- **Research-Driven Development:** Insights and research guiding the development process are documented inside [this file](docs/research.md) and [this one](docs/ideas.md).

---

## Getting Started

### Prerequisites
1. **Python Environment:** Install Python 3.11+.
2. **Pytroch:** Install [Pytorch](https://pytorch.org/get-started/locally/).
3. **Dependencies:** Install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```
4. **Docker(Optional, but recommend):** Ensure Docker is installed for deployment.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/LMicol/instrument-classifier
   ```
2. Navigate to the project directory:
   ```bash
   cd instrument-classifier/
   ```
3. Set up the environment and install dependencies.

---

## Usage

### Instruments sounds Dataset
The dataset used in this project can be found at [Micol/musical-instruments-sound-dataset](https://huggingface.co/datasets/Micol/musical-instruments-sound-dataset).

I've used this [Kaggle dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset/) as base and made some changes using the scripts in `src/helpers`.

### Training Models
Explore the `src/models` directory for Jupyter Notebooks to train and evaluate models. Each model has its corresponding training script and saved weights.
To train a model in your machine or play with the notebooks, you will need to setup the whole environment.

**I recommend using docker and the web view if you just want to test the final model.**

### Web view
In the folder `src/web` you'll find a simple web interface to test the model API with your microfone, I recommend using Firefox to test it.
If you allow your microphone, the response from the model will be highlighted with a red box.
If you want to use a file upload, the highlight will be a blue box and won't change.

### Deployment
#### Using Docker Compose
For deployment of both the server and web interface, use the `docker-compose.yml` file provided in the repository. This will set up two services:
1. **Web Interface**: Runs on port `5000`.
2. **Audio Server**: Runs on port `8000`.

To deploy, run the following command in the project root directory:
```bash
docker-compose up --build
```
Access the services:
- Web Interface: `http://localhost:5000`
- API Server: `http://localhost:8000`

Access the web interface at `http://localhost:5000`.

---

## Documentation

### Research
The `docs/research.md` file contains detailed information about the research conducted to guide model development.
The file is structured in three categories:
- Personal Thoughts: Personal monologues and internal discussion I've had
- Actions: Things I've done.
- Research Oobservations: Comments about code, model behavior and, research overall. 

### Ideas
The `docs/ideas.md` file includes:
- How the model was developed and idea behind the implementation.
- Results for each model iteration.
- Confusion matrices and performance insights.

### Visuals
The `images` directory contains performance metrics for each model.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
