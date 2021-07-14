# PIMAnalyzer: Tool for analyzing the details of crossbar for different applications on PIM

PIMAnalyzer is a tool based on pytorch. 
It models the crossbars and the peripheral circuits on PIM.
The applications on PIM (e.g., the inference of neural networks) can be mapped on the crossbars.
It simulates the computations on these crossbars given the input dataset.
The tool can generate the distribution of the input data in each row, each column and each cell in crossbar.
Analyzing these distributions is expected to be benefit to the future design of PIM hardware.

# Install

Requirement 
- python>=3.5
- pytorch>=1.5
- matplotlib
- pandas

# Usage

### Analyze neural networks

Example of analyze ResNet-18 on ImageNet dataset.

At first, put your ImageNet dataset into folder `data/imagenet`.
Then run this command:
```
python3 analyze_nn.py resnet18 imagenet --data_root <path-to-data>
```

The distribution of each layer, and the distribution of all layers is printed.