# Zero-shot and Few-shot Classification with LLMs

## 3a: Ollama generation parameters

The model used is gemma3, served via Ollama, running locally. It makes sense to run this locally, as development takes place on a powerful computer, and there is no dependency on an external API. Gemma3 was chosen because it is a light, open-weight model that has proven to perform reasonably well. It was developed by Google DeepMind and has 4.3 Billion parameters. The local version is 4-bit quantized and therefore uses only around 3GB of disk space. Details on how to set it up can be found in the `appendix.md`.  
The generation parameters `GEN_PARAMS = {"num_predict": 10, "temperature": 0, "seed": 26}` were used.  
`"num_predict": 10` : This limits the length of the output. As only need a party name is needed, 10 should be enough.
`"temperature": 0` : For a deterministic output where the seed can be set, temperature needs to be set to 0.
`"seed": 26` : This number can be an arbitrary value, setting it fixes the randomness sampling so that runs are deterministic and therefore reproducible. 26 is reused to be consistent with the train/test split in part two.
