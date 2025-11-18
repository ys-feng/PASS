# Experiments
Below are the instructions for running experiments using our novel ChestAgentBench and the previous SoTA CheXbench. ChestAgentBench is a comprehensive benchmark containing over 2,500 complex medical queries across 8 diverse categories.

### ChestAgentBench

To run gpt-4o on ChestAgentBench, enter the `experiments` directory and run the following script:
```bash
python benchmark_gpt4o.py
```

To run llama 3.2 vision 90B on ChestAgentBench, run the following:
```bash
python benchmark_llama.py
```

To run chexagent on ChestAgentBench, run the following:
```bash
python benchmark_chexagent.py
```

To run llava-med on ChestAgentBench, you'll need to clone their repo and copy the following script into it, after you follow their setup instructions.
```bash
mv benchmark_llavamed.py ~/LLaVA-Med/llava/serve
python -m llava.serve.benchmark_llavamed --model-name llava-med-v1.5-mistral-7b --controller http://localhost:10000
```

If you want to inspect the logs, you can run the following. It will select the most recent log file by default.
```bash
python inspect_logs.py [optional: log-file] -n [num-logs]
```

Finally, to analyze results, run:
```bash
python analyze_axes.py results/[logfile].json ../benchmark/questions/ --model [gpt4|llama|chexagent|llava-med] --max-questions [optional:int]
```

### CheXbench

To run the models on chexbench, you can use `chexbench_gpt4.py` as a reference. You'll need to download the dataset files locally, and upload them for each request. Rad-ReStruct and Open-I use the same set of images, so you can download the `NLMCXR.zip` file just once and copy the images to both directories.

You can find the datasets here:
1. [SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering](https://www.med-vqa.com/slake/). Save this to `MedMAX/data/slake`.
2. [Rad-ReStruct: A Novel VQA Benchmark and Method for Structured Radiology Reporting](https://github.com/ChantalMP/Rad-ReStruct). Save the images to `MedMAX/data/rad-restruct/images`.
3. [Open-I Service of the National Library of Medicine](https://openi.nlm.nih.gov/faq). Save the images to `MedMAX/data/openi/images`.

Once you're finished, you'll want to fix the paths in the `chexbench.json` file to your local paths using the `MedMax/data/fix_chexbench.py` script.


### Compare Runs
Analyze a single file based on overall accuracy and along different axes
```
python compare_runs.py results/medmax.json
```

For a direct evaluation comparing **2** models, on the exact same questions 
```
python compare_runs.py results/medmax.json results/gpt4o.json
```

For a direct evaluation comparing **ALL** models, on the exact same questions (add as many model log files as you want).
```
python compare_runs.py results/medmax.json results/gpt4o.json results/llama.json results/chexagent.json results/llavamed.json
```
