import chika
import torch
import torch.nn.functional as F
from homura import lr_scheduler, optim, reporters, trainers
from homura.vision import DATASET_REGISTRY

from models import EMA, MODEL_REGISTRY


@chika.config
class Config:
    name: str = chika.choices(*MODEL_REGISTRY.choices())
    batch_size: int = 128

    epochs: int = 200
    lr: float = 0.1
    weight_decay: float = 1e-4

    use_ema: bool = False

    bn_no_wd: bool = False
    use_amp: bool = False
    use_prefetcher: bool = False
    debug: bool = False


@chika.main(cfg_cls=Config)
def main(cfg):
    model = MODEL_REGISTRY(cfg.name)(num_classes=10)
    model(torch.randn(1, 3, 32, 32))
    if cfg.use_ema:
        model = EMA(model)
    # to infer shape...
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(cfg.batch_size, num_workers=4,
                                                            use_prefetcher=cfg.use_prefetcher)
    optimizer = None if cfg.bn_no_wd else optim.SGD(lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.epochs, 4, 5)

    with trainers.SupervisedTrainer(model,
                                    optimizer,
                                    F.cross_entropy,
                                    reporters=[reporters.TensorboardReporter('.')],
                                    scheduler=scheduler,
                                    use_amp=cfg.use_amp,
                                    debug=cfg.debug
                                    ) as trainer:
        for _ in trainer.epoch_range(cfg.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    main()
