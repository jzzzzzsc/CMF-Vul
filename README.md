# CMF-Vul

The pre-trained models used in this project can be obtained from the following links:

- **UniXcoder**  
  https://huggingface.co/microsoft/unixcoder-base

- **CodeBERT**  
  https://huggingface.co/microsoft/codebert-base

- **SwinV2**  
  https://huggingface.co/docs/transformers/model_doc/swinv2

### Prerequisites
- Python 3.8+
- Joern (version 1.1.995 to 1.1.1125)
- CUDA 11.3 (for GPU acceleration)

###1.PDG Generation

Install Joern (version 1.1.995 to 1.1.1125 recommended)
```bash

# Generate .bin files
python joern_graph_gen.py -i ./data/sard -o ./data/sard/bins -t parse


# Generate PDGs (.dot files)
python joern_graph_gen.py -i ./data/sard/bins -o ./data/sard/pdgs -t export -r pdg

```
