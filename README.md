# NanoKGAT
**Analysis-and-evaluation-of-GNN-algorithms-on-multiple-antibody-binding-site-datasets**

Nanobodies are artificial antibodies derived from the immune systems of camelid animals, obtained through artificial processing and isolation of antigen-binding proteins. Their small molecular size and high specificity endow nanobodies with broad potential applications across various fields. However, determining the paratope (antigen-binding region) of nanobodies through experimental methods is both expensive and time-consuming, and traditional computational methods often lack sufficient accuracy. To improve prediction accuracy, we propose NanoKGAT, a method that employs Graph Neural Networks (GNN) and the antibody pre-trained language model AntiBERTy to fully leverage the sequence and three-dimensional structural information of nanobodies.specifically, our method first utilizes AntiBERTy to extract high-level sequence features from the nanobody sequences. These features capture the complex dependencies between amino acid residues. Simultaneously, we extract spatial features from the three-dimensional structure of the nanobody, construct a graph structure, and employ two graph convolution mechanisms: EdgeConv and GATConv for feature learning. EdgeConv enhances the edge information between nodes, while GATConv dynamically adjusts the weights of different residues through an attention mechanism, thereby further improving the prediction accuracy of the model.in terms of model architecture, we incorporate a one-dimensional convolutional neural network to process the sequence features and fuse them with the output of the graph convolution layers to form a comprehensive feature representation. Finally, through a fully connected layer and a sigmoid activation function, the model outputs the probability of each amino acid residue belonging to the paratope. Our approach relies solely on the sequence and structural information of the nanobody itself, without the need for additional antigen data, significantly simplifying the prediction process. 

# Install

**Clone the repo**

```
[git clone https://github.com/Wo-oh-oh-ooh-oh/Nanotope](https://github.com/moppppp/Analysis-and-evaluation-of-GNN-algorithms-on-multiple-antibody-binding-site-datasets)

```

**Create a virtual env**

```
conda create --name Nanotope python=3.9
```

**install**

```
pip install .
```

