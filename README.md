# Python script implementation of Residual 50/101/152-layer Network with Identity Mapping
<img src="http://7xrja7.com1.z0.glb.clouddn.com/identity_mapping_resnet.png" alt="residual net structure image" width="200px" /></br>
the new net structure is given in this figure, in which we can see that **BN** layer and **ReLU** layer are stacked before weight layer. This script provide simple way to choose the layer to generate:</br>
<pre>
layer_50_layer = (3, 4, 6, 3)
layer_101_layer = (3, 4,  23, 3)
layer_152_layer = (3, 8, 38, 3)
</pre>
