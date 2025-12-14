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
