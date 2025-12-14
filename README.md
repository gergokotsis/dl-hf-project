# dl-hf-project
Deep learning házi feladat projekt

A projekt futtatásához szükséges megadni egy /output és egy /data mappát. A data mappában a "legaltextdecoder.zip" fájlnak kell szerepelnie. A futtatás a docker run --rm --gpus all \
 -v C:\eleresi\ut\az\adatokhoz:/data \
 -v C:\eleresi\ut\az\outputhoz:/app/output \
 my-dl-project-work-app:1.0 > training_log.txt 2>&1
parancs segítségével lehetséges.
Az eredmények a training_log.txt fájlban, és az output mappa evaluation_report.txt és evaluation_summary.csv fájljaiban található meg. Az inkrementális modellfejlesztés lépései a notebook mappában találhatóak egy jupyter notebookban.
