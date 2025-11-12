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
