# Deep Learning Class (VITMMA19)

## Project Levels
Basic Level (for signature)

## Data preparation: 

The 01-data-preprocessing.py script uses the given link to download the data (legaltextdecoder.txt) from Microsoft Sharepoint. The script finds the json files containing the data. The consensus folder is excluded. The data is extracted from the json files. After this the script does minor data cleaning, and checks for duplicates. After some analysis it saves the created dataset. The data is split into training, validation and test sets in the training script.

## Project Details
### Project Information
* Selected Topic: Legal Text Decoder
* Student Name: Kotsis Gergő
* Aiming for +1 Mark: No
### Solution Description:

The task is a classification problem, where we have to decide how hard it is to understand a given legal text (ÁSZF) snippet. I solved the problem with an NLP solution. During the model development process I started by creating a simple TF-IDF + MLP model, that is my baseline model. After this I tried different technologies to make the model better. Tried technologies: Bag of Words, LSTM, GRU, Embedding, Dropout layers, L2 regularization, weighting the training data (imbalaced dataset). The final model is has an embedding layer that is followed by an MLP. This model gave me the best results. The final results show, that the final model is about 9% more accurate then the baseline model, that I started with, and 19% better, then guessing randomly. The final pipeline downloads the data from the internet, and does basic data preparation on it. After this a basline and the final model are trained on the same data, adn then evaluated on the same data. Finally on two models are used for inference. Two texts are given to each, and they have to return how easy it is to understand the given text.

## Docker Instructions
This project is containerized using Docker. Follow the instructions below to build and run the solution.

### Build
Run the following command in the root directory of the repository to build the Docker image:
```bash
docker build -t dl-project .
```
### Run
To run the solution, use the following command:
```bash
docker run dl-project > log/run.log 2>&1
```
The > log/run.log 2>&1 part ensures that all output (standard output and errors) is saved to log/run.log.

If it is necessary to have the trained models, after running the container, you can run the container with this command:
```bash
docker run -v /absolute/path/to/local/output:/app/output dl-project > log/run.log 2>&1
```
With this command the models will be saved to this folder: /absolute/path/to/local/output Replace /absolute/path/to/your/local/output with the actual path to the folder, where you want the models to be saved.

To capture the logs for submission (required), redirect the output to a file:

```bash
docker run dl-project > log/run.log 2>&1
```
The container is configured to run every step (data preprocessing, training, evaluation, inference).

## File Structure and Functions

The repository is structured as follows:

* **src/:** Contains the source code for the machine learning pipeline.

    * 01-data-preprocessing.py: Script for loading, cleaning, and preprocessing the raw data.
    * 02-training.py: The main script for spliting the data, defining the model and executing the training loop.
    * 03-evaluation.py: Script for evaluating the trained baseline model and final model on test data and generating metrics.
    * 04-inference.py: Script for running the model on new, unseen data to generate predictions.
    * config.py: Configuration file containing hyperparameters (e.g., epochs) and paths.
    * utils.py: Helper functions and utilities used across different scripts.
* notebook/: Contains Jupyter notebook for experimentation.

    * model_development.ipynb: Jupyter notebook containing the experimentation, that led to the final model.

* log/: Contains log files.

    * run.log: Example log file showing the output of a successful training run. When the program is run with the command, that captures logs, the new logs override the old logs. (The current run.log file contains warnings, because it did not found a GPU, it should be fine when run on a container, that can use a GPU.)
* Root Directory:

    * Dockerfile: Configuration file for building the Docker image with the necessary environment and dependencies.
    * requirements.txt: List of Python dependencies required for the project.
    * README.md: Project documentation and instructions.
