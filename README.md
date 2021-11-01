## Can contrastive learning avoid shortcut solutions?

The generalization of representations learned via contrastive learning depends crucially on  what features of the data are extracted. However, we observe that the contrastive loss does not always sufficiently guide which features are extracted, a behavior that can negatively impact the performance on downstream tasks via "shortcuts", i.e., by inadvertently suppressing important predictive features. We find that feature extraction is influenced by the difficulty of the so-called instance discrimination task (i.e., the task of discriminating pairs of similar points from pairs of dissimilar ones). Although harder pairs improve the representation of some features, the improvement comes at the cost of suppressing previously well represented features. In response, we propose implicit feature modification (IFM), a method for altering positive and negative samples in order to guide contrastive models towards capturing a wider variety of predictive features. Empirically, we observe that IFM reduces feature suppression, and as a result improves performance on vision and medical imaging tasks.

<p align='center'>
<img src='https://github.com/joshr17/IFM/blob/main/fig1.png?raw=true' width='800'/>
</p>

An ideal encoder would  discriminate between instances using multiple distinguishing features instead of finding simple shortcuts that suppress features. We show that InfoNCE-trained encoders can suppress features (Sec. 2.2). However, making instance discrimination harder during training can trade off representation of different features (Sec. 2.3). To avoid the need for trade-offs we propose implicit feature modification (Sec. 3), which reduces suppression in general, and improves generalization (Sec. 4). 

**Can contrastive learning avoid shortcut solutions?** [[paper]](https://arxiv.org/pdf/2106.11230.pdf)
<br/>
[Joshua Robinson](https://joshrobinson.mit.edu/), 
[Li Sun](https://lisun97.github.io/), 
[Ke Yu](http://www.isp.pitt.edu/node/1945),
[Kayhan Batmanghelich](https://batman-lab.com/), 
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/), and
[Suvrit Sra](http://optml.mit.edu/)
<br/>

**Implicit feature modification**

In this paper we present implicit feature modification, a method for reducing shortcut learning in contorstive leanring while adding no computational overhead, and requiring only a couple of lines of code to implement. We also find that IFM improves downstream generalization. This repo contains a minimally modificed version of the official [MoCo code](https://github.com/facebookresearch/moco) to illustrate the simplicity of the implementation. 

To reproduce our ImageNet100 results, first Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet), and select the [100 sublasses](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt). To do 200 epochs of unsupervised pre-training using a ResNet-50 model using our method, run for example:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.8 \
  --batch-size 512 \
  --moco-m 0.99 \
  --mlp --moco-t 0.2 --aug-plus --cos
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --method ifm \
  --epsilon 0.1 \
  --dataset_root PATH/TO/DATA
```

To run standard MoCo-v2 simply remove the `--method` and `--epsilon` arguments. Training should fit on any 8-gpu machine, but also works on 4 Tesla V100s. For linear evaluation run,

```
python main_lincls.py \
--pretrained=model_best.pth.tar \
--lr 10.0  \ 
-b=128  \
--schedule 30 40 50 \
--epochs 60 \
--dist-url=tcp://localhost:10001 \
--dataset_root=PATH/TO/DATA
```

Checkpoints for models pre-trained on ImageNet100 can be downloaded here:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">epsilon</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">checkpoint</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo-v2</a></td>
  <td align="center">200</td>
<td align="center">0</td>
<td align="center">80.5</td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left"><a href="">IFM-MoCo-v2</a></td>
  <td align="center">200</td>
<td align="center">0.05</td>
<td align="center">81.1</td>
<td align="center"><a href="">download</a></td>
</tr>
<tr><td align="left"><a href="">IFM-MoCo-v2</a></td>
  <td align="center">200</td>
<td align="center">0.1</td>
<td align="center">81.4</td>
<td align="center"><a href="">download</a></td>
</tr>
  <tr><td align="left"><a href="">IFM-MoCo-v2</a></td>
    <td align="center">200</td>
<td align="center">0.2</td>
<td align="center">80.9</td>
<td align="center"><a href="">download</a></td>
</tr>
  
</tbody></table>

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{robinson2021shortcuts,
  title={Can contrastive learning avoid shortcut solutions?},
  author={Robinson, Joshua and Sun, Li and Yu, Ke and Batmanghelich, Kayhan and Jegelka, Stefanie and Sra, Suvrit},
  journal={NeurIPS},
  year={2021}
}
```
For any questions, please contact Josh Robinson (joshrob@mit.edu).
