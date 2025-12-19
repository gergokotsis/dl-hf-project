# dl-hf-project
Deep learning házi feladat projekt

Project Details
Project Information
Selected Topic: Legal Text Decoder
Student Name: Kotsis Gergő
Aiming for +1 Mark: No
Solution Description
A feladat megoldása során egy olyan modellt akartam elkészíeni, amei minnál jobban képes megoldani a bonyolult jogi szövegek osztályozását. Mivel az adathalmaz elég kicsi volt és a cimkézés sem volt tökéletes egyszerűbb NLP megoldásokkal próbálkoztam, amelyek a feltöltött notewbookban szerepelnek.

Docker Instructions
This project is containerized using Docker. Follow the instructions below to build and run the solution. [Adjust the commands that show how do build your container and run it with log output.]

Build

A projekt futtatásához szükséges megadni egy /output és egy /data mappát. A data mappában a "legaltextdecoder.zip" fájlnak kell szerepelnie. A build parancs: docker build -t dl-project . A futtatás a docker run --rm --gpus all \
 -v C:\eleresi\ut\az\adatokhoz:/data \
 -v C:\eleresi\ut\az\outputhoz:/app/output \
 my-dl-project-work-app:1.0 > training_log.txt 2>&1
parancs segítségével lehetséges.
Az eredmények a training_log.txt fájlban, és az output mappa evaluation_report.txt és evaluation_summary.csv fájljaiban található meg. Az inkrementális modellfejlesztés lépései a notebook mappában találhatóak egy jupyter notebookban.
File Structure and Functions

The repository is structured as follows:

src/: Contains the source code for the machine learning pipeline.

01-data-preprocessing.py: Adatok betöltése, előfeldolgozása
02-training.py: Baseline modell betanítása, végső modell betanítása
03-evaluation.py: Modellek kiértékelése, eredmények mentése
notebook/: Feladathoz használt notebook mappája

model_development.ipynb: Adatok elemnzése, modelltanítási kísérletek

Root Directory:

Dockerfile: Configuration file for building the Docker image with the necessary environment and dependencies.
requirements.txt: List of Python dependencies required for the project.
README.md: Project documentation and instructions.
A feladat megoldásában igénybe vettem LLM-ek segítségét. (ChatGPT, Gemini)

Deep Learning Class (VITMMA19)

Submission Instructions

Project Levels
Basic Level (for signature)

Data preparation: 

The 01-data-preprocessing.py script expects the legaltextdecoder.zip file to be in the /data folder. It extracts the zip, and then in the folder looks for other folders that should contain the json files with the data. The script loads the data where it finds a label for it. It does not load the data from the consensus file, so it will not duplicate data.

Logging Requirements
The training process must produce a log file that captures the following essential information for grading:

Configuration: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
Data Processing: Confirm successful data loading and preprocessing steps.
Model Architecture: A summary of the model structure with the number of parameters (trainable and non-trainable).
Training Progress: Log the loss and accuracy (or other relevant metrics) for each epoch.
Validation: Log validation metrics at the end of each epoch or at specified intervals.
Final Evaluation: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).
The log file must be uploaded to log/run.log to the repository. The logs must be easy to understand and self explanatory. Ensure that src/utils.py is used to configure the logger so that output is directed to stdout (which Docker captures).


 Solution Description: Provided a clear description of your solution, model, and methodology.
 Configuration: Used src/config.py for hyperparameters and paths, contains at least the number of epochs configuration variable.
 Logging:
 Log uploaded to log/run.log
 Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.



Project Details
Project Information
Selected Topic: Legal Text Decoder
Student Name: Kotsis Gergő
Aiming for +1 Mark: No
Solution Description
[Provide a short textual description of the solution here. Explain the problem, the model architecture chosen, the training methodology, and the results.]

Docker Instructions
This project is containerized using Docker. Follow the instructions below to build and run the solution.

Build
Run the following command in the root directory of the repository to build the Docker image:

docker build -t dl-project .
Run
To run the solution, use the following command. You must mount your local data directory to /app/data inside the container.

To capture the logs for submission (required), redirect the output to a file:

docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
Replace /absolute/path/to/your/local/data with the actual path to your dataset on your host machine that meets the Data preparation requirements.
The > log/run.log 2>&1 part ensures that all output (standard output and errors) is saved to log/run.log.
The container is configured to run every step (data preprocessing, training, evaluation, inference).
File Structure and Functions

The repository is structured as follows:

src/: Contains the source code for the machine learning pipeline.

01-data-preprocessing.py: Scripts for loading, cleaning, and preprocessing the raw data.
02-training.py: The main script for defining the model and executing the training loop.
03-evaluation.py: Scripts for evaluating the trained model on test data and generating metrics.
04-inference.py: Script for running the model on new, unseen data to generate predictions.
config.py: Configuration file containing hyperparameters (e.g., epochs) and paths.
utils.py: Helper functions and utilities used across different scripts.
notebook/: Contains Jupyter notebooks for analysis and experimentation.

01-data-exploration.ipynb: Notebook for initial exploratory data analysis (EDA) and visualization.
02-label-analysis.ipynb: Notebook for analyzing the distribution and properties of the target labels.
log/: Contains log files.

run.log: Example log file showing the output of a successful training run.
Root Directory:

Dockerfile: Configuration file for building the Docker image with the necessary environment and dependencies.
requirements.txt: List of Python dependencies required for the project.
README.md: Project documentation and instructions.