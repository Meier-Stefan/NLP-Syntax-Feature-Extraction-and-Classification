# Zero-shot and Few-shot Classification with LLMs

## 3a: Ollama generation parameters

The model used is gemma3, served via Ollama, running locally. It makes sense to run this locally, as development takes place on a powerful computer, and there is no dependency on an external API. Gemma3 was chosen because it is a light, open-weight model that has proven to perform reasonably well. It was developed by Google DeepMind and has 4.3 Billion parameters. The local version is 4-bit quantized and therefore uses only around 3GB of disk space. Details on how to set it up can be found in the `appendix.md`.  
The generation parameters `GEN_PARAMS = {"num_predict": 10, "temperature": 0, "seed": 26}` were used.  
`"num_predict": 10` : This limits the length of the output. As only need a party name is needed, 10 should be enough.
`"temperature": 0` : For a deterministic output where the seed can be set, temperature needs to be set to 0.
`"seed": 26` : This number can be an arbitrary value, setting it fixes the randomness sampling so that runs are deterministic and therefore reproducible. 26 is reused to be consistent with the train/test split in part two.

## 3d: Comparison between Zero-Shot and Few-Shot

### Overall result

Across all experiments, the zero-shot setup with Gemma3 achieved the highest performance, with a macro F1 score of 0.8697, and consistently outperformed every prior classifier as well as every few-shot configuration tested. 

### Methods

Because the textbook examples are very short, and a full Hansard speech would not fit comfortably into the model’s context window, each training example speech was truncated to a fixed character length before being included in the prompt as a “speech start”. Small prompt wording changes during development prior to the experiment turned out to matter: changing the instruction from “here are example speeches” to “Here are example speech starts” led to a noticeable improvement in performance, illustrating how minor edits to the instructions can affect model behaviour.  
As experiment to explore few-shot prompting, two main dimensions were varied: the number of example speech starts per party (one vs two) and the truncation length of each example (400, 500, 600, 700, and 800 characters). For each combination of these settings, three different random seeds were used to sample speeches from the training set, so that it was possible to see how sensitive the model was to which specific examples were chosen. In total, this gave 5 lengths × 3 seeds × 2 example counts = 30 individual few-shot runs.

## Few-shot results

Across the three seeds, no single seed produced the best macro F1 scores consistently across all lengths and example counts; performance varied depending on both how many examples were used and how long they were. The single best few-shot run was obtained with two example speech starts per party and a truncation length of 400 characters, and achieved a macro F1 score of 0.6556.  

When aggregating over seeds by averaging the macro F1 scores obtained with a given configuration, the three strongest configurations were:  

400 characters with one example per party (average macro F1 ≈ 0.5375)  
700 characters with one example per party (average macro F1 ≈ 0.5369)  
700 characters with two examples per party (average macro F1 ≈ 0.5203)  

Given that these averages are very close, and that the zero-shot baseline is much higher, a reasonable choice, if a few-shot setup had to be deployed, would be one example per party with 400-character speech starts. This configuration is among the best-performing few-shot options while also using a shorter prompt, which reduces token usage, speeds up inference, and lowers computational and energy costs.  
That said, for another experiment the `qwen3:14b` model was available on this machine. Curiosity has triumphed over reason - the outputs of the zero-shot and few-shot with one example per party with 400-character speech starts can be found in the `appendix.md` file. The script took 52 min 37 sec to run and outperformed gemma3 on zero-shot with macro F1 0.93 vs 0.87 and is also notably better on few-shot, the best macro F1 score was 0.78.
