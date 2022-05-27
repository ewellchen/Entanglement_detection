# Detecting quantum entanglement with unsupervised learning

A Pytorch implementation of paper https://arxiv.org/abs/2103.04804.

**Abstract:**
Quantum properties, such as entanglement and coherence, are indispensable resources in various quantum information processing tasks. However, there still lacks an efficient and scalable way to detecting these useful features especially for high-dimensional quantum systems. In this work, we exploit the convexity of normal samples without quantum features and design an unsupervised machine learning method to detect the presence of quantum features as anomalies. Particularly, given the task of entanglement detection, we propose a complex-valued neural network composed of pseudo-siamese network and generative adversarial net, and then train it with only separable states to construct non-linear witnesses for entanglement. It is shown via numerical examples, ranging from 2-qubit to 10-qubit systems, that our network is able to achieve high detection accuracy with above 97.5% on average. Moreover, it is capable of revealing rich structures of entanglement, such as partial entanglement among subsystems. Our results are readily applicable to the detection of other quantum resources such as Bell nonlocality and steerability, indicating that our work could provide a powerful tool to extract quantum features hidden in high-dimensional quantum data.

## Prerequisite

The main requirements can be intalled by:

```
cd 2-qubit
pip install -re requirements.txt
```

## Training

```
cd 2-qubit
python main.py
```
## License

MIT License

## Citation

If you find our work useful in your research, please consider citing:

```
@article{Chen_2021,
	doi = {10.1088/2058-9565/ac310f},
	url = {https://doi.org/10.1088/2058-9565/ac310f},
	year = 2021,
	month = {nov},
	publisher = {{IOP} Publishing},
	volume = {7},
	number = {1},
	pages = {015005},
	author = {Yiwei Chen and Yu Pan and Guofeng Zhang and Shuming Cheng},
	title = {Detecting quantum entanglement with unsupervised learning},
	journal = {Quantum Science and Technology},
}
```



