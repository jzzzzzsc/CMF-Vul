# CMF-Vul
## Data Availability

The datasets used and analyzed in this study are publicly available at the following locations:

- **Big-Vul**  
  https://github.com/ZeoVan/MSR_20_Code_vulnera-bility_CSV_Dataset

- **SARD**  
  https://github.com/Lifeasarain/MGVD/blob/master-/dataset/sard/csv/data.csv

- **FFmpeg & Qemu**  
  https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit
  
## Pre-trained models
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
###2.PDG images Generation
The PDG images in this project are generated using **Xdot**.
We note that the same functionality can also be achieved via the
`graphviz` library in Python.

To convert graph representations into image files, please run the
provided script:

```bash
python graph-image.py

