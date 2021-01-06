# attentions.pytorch

**🚧WORK IN PROGRESS🚧**

## Requirements

* Python>=3.8
* PyTorch>=1.7.1
* homura-core>=2020.12.0
* chika
* opt_einsum (optional)

```commandline
conda create -n attentions python=3.8
conda install -c pytorch pytorch torchvision cudatoolkit=11.0
pip install -U homura-core chika [opt_einsum]
```

## ΛNet on CIFAR-10

```commandline
python lambdanet_cifar.py --name [lambda_resnet20,efficient_resnet20,dotprod_resnet20] [--use_ema]
```

It would produce NaNs.