# how to run

```bash
./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/data/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results/ http://kumo01.tsc.uc3m.es:11434
```

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/data/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_no_limit/ http://kumo01.tsc.uc3m.es:11434

For tests (sample of 75 with failures cases in the first implementation)

```bash
./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/PLACE/temporal/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results/ http://kumo01.tsc.uc3m.es:11434

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/PLACE/temporal2/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results/ http://kumo01.tsc.uc3m.es:11434

## IMPORTANT 

Remove test files

Add openai key!!

```bash
./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/PLACE/to_process_cpv8/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_cpv8/ http://kumo01.tsc.uc3m.es:11434
```

```bash
./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/PLACE/to_process_cpv5/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_cpv5/ http://kumo01.tsc.uc3m.es:11434
```


## RUN FOR ALL

### outsiders

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_outsiders_2024_conTitleCPVLink_chunks/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_outsiders_2024/ http://kumo01.tsc.uc3m.es:11434
--> DONE 


### insiders

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_conTitleCPVLink_chunks/part_00/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_00/ http://kumo01.tsc.uc3m.es:11434
--> DONE 

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_conTitleCPVLink_chunks/part_01/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_01/ http://kumo01.tsc.uc3m.es:11434

./experiments/objective_extractor/run_extractor.sh /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_conTitleCPVLink_chunks/part_02/ /export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_02/ http://kumo01.tsc.uc3m.es:11434