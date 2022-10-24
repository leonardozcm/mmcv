# MMCV Estimator Design

Our mmcv estimator design basically follows Orca torch estimator design.

## MMCV Usage Example

The core components of mmcv are Runners and Hooks.


### Runners
A runner usually contains all kinds of components(model, optimizer, scheduler, hooks etc., except dataloader) and intermediate output(data_batch, loss etc.) during the training loop.
```python
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=logger,
        max_epochs=4)
``` 
You can access these components directly like `self.model`, and mmcv encourages you to do this to achieve customized training. Especially, **Model(torch.nn.module)** should implement additional functions `    def train_step(self, data, optimizer):` and `    def val_step(self, data, optimizer):` for runners to call.

There are two kinds of runner in mmcv: EpochBasedRunner and IterBasedRunner, and their use cases are aptly named. Our design and development is based on most used **EpochBasedRunner**. There are four APIs all runner should implement:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

`run()` will call `train` and `val` frequently to manage training and evaluating intervals.

Another necessary action is to register hooks to `runner.hooks`. There are hooks provided by mmcv and user hooks, and we talk about this later. For `mmcv.hooks` you can simply just pass configs:
```python
    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # configuration of saving checkpoints periodically
    checkpoint_config = dict(interval=1)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config)

    # register other hooks
    eval_hook = EvalHook(valloader, interval=1, test_fn=cifar10_test_fn, greater_keys=["acc"])
    runner.register_hook(eval_hook)
```
Only `optimizerHook` is required for backward stuffs.

To start training:
```python
    # train 3 epochs and val once, and then so on.
    runner.run([train_dataloader, val_dataloader], [('train', 3), ('val', 1)])
```

### Hooks
A Hook is declared as multiple stages like:
```python
class Hook:
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)
    ...
```
They all take `runner` as the only arg, so users can access almost all the variables they need and implement their own logic:
```python
class OptimizerHook(Hook):
    ...
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward() # this is why optimizerHook is required

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

```
Runners will call these hooks at the right time:
```python

class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
```

## MMCV Estimator Design
Different from PytorchRayEstimator, there are only `runner_creator(config)-->mmcv.EpochBasedRunner` and `dataloader_creator(config)-->Dataloader` should pass to MMCVEstimator.

### User API
1. Init API
```python
    def runner_creator(config):
        model = Model()

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        logger = get_logger('mmcv')
        # runner is a scheduler to manage the training
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir='./work_dir',
            logger=logger,
            max_epochs=4)

        # learning rate scheduler config
        lr_config = dict(policy='step', step=[2, 3])
        # configuration of optimizer
        optimizer_config = dict(grad_clip=None)
        # configuration of saving checkpoints periodically
        checkpoint_config = dict(interval=1)
        # save log periodically and multiple hooks can be used simultaneously
        log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
        # register hooks to runner and those hooks will be invoked automatically
        runner.register_training_hooks(
            lr_config=lr_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config)

        return runner

    def train_dataloader_creator(config):
        # dataset and dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = CIFAR10(
            root='data', train=True, download=True, transform=transform)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        return trainloader

    est = MMCVEstimator(runner_creator=runner_creator,config={})
```

2. Fit API
Same as runner.run()
```python
    est.fit([train_dataloader_creator, val_dataloader_creator], [('train', 1), ('val', 1)])
```

### Internal API
1. MMCVEstimator
For ray backend we just replace TorchRunner with MMCVRunner, and others keep the same:
```python
class MMCVEstimator(OrcaRayEstimator):
    def __init__(self, runner_creator=None, config=None) -> None:
        super().__init__()
        ...
        RemoteRunner = ray.remote(num_cpus=cores_per_node)(MMCVRunner)

        params = dict(runner_creator = self.runner_creator,
                      config=config)

        self.remote_workers = [
            RemoteRunner.remote(**params) for _ in range(num_nodes)
        ]
```
For est.fit():
```python
    def fit(self,
            data_loaders:List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None, # deprecated
            **kwargs) -> None:
        params=dict(data_loaders_creators=data_loaders,
                    workflow=workflow,
                    max_epochs=max_epochs,
                    **kwargs)
        success, worker_stats = self._train_epochs(**params)
```

2. MMCVRunner
We inherit MMCVRunner from `EpochBasedRunner`, and expose attributes of runner from `runner_creator` to MMCVRunner itself for concise(discuss later):

```python
class MMCVRunner(EpochBasedRunner):
    ...
    # Expose attrs of EBR here for concise when overriding `train_step` etc. in EpochBasedRunner,
    # for example:
    #    before: self.runnner.model(batch)
    #    now: self.model(batch)
    def _wrapFromEBR(self, ebrunner):
        for k in self.EBR_slots:
            # todo: check neccessary components
            setattr(self, k, getattr(ebrunner, k))
    
    def setup_components(self):
        runner = self.runner_creator(self.config)

        self._wrapFromEBR(runner)
```
There are benefits to do this, when overriding run(), train(), val() etc. sticked in EpochBasedRunner with our own logic.
```python
    # ray remote api
    def run(self,
            data_loaders_creators:List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None, # deprecated
            **kwargs) -> None:

        data_loaders = [self.with_sampler(dataloader_creator(self.config)) for dataloader_creator in data_loaders_creators]

        super().run(data_loaders, workflow, max_epochs, **kwargs)
        # instead of 
        # super().runner.run(data_loaders, workflow, max_epochs, **kwargs)
```
Its more natural to our developers and prevent typos. And other setup processes remain the same with TorchRunner.
```python
    def setup_torch_distribute(self, tcp_store_host, tcp_store_port, world_rank,
                               world_size):
        import torch.distributed as dist
        from mmcv.parallel.distributed import MMDistributedDataParallel

        client_store = dist.TCPStore(tcp_store_host, tcp_store_port, -1, False)        
        dist.init_process_group(
            backend="gloo",
            store=client_store,
            rank=world_rank,
            world_size=world_size)
        self.backend = "torch-distributed"
        self.world_rank = world_rank
        self.size = world_size
        self.setup_components()
        
        #     - It supports a custom type :class:`DataContainer` which allows more
        #       flexible control of input data.
        #     - It implement two APIs ``train_step()`` and ``val_step()``.
        self.model = MMDistributedDataParallel(self.model) # runner.model: `torch.nn.Module`
```

## Consistency Evaluation Status

- [X] Local Runnable
- [X] Local Convergence
- [X] Dist Runnable
- [] Dist Runnable(pending)

```
(MMCVRunner pid=21452) 2022-10-24 20:26:13,353 - mmcv - INFO - Epoch(val) [1][313]      acc: 0.3038
(MMCVRunner pid=21452) 2022-10-24 20:26:54,456 - mmcv - INFO - Epoch(val) [2][313]      acc: 0.4156
(MMCVRunner pid=21452) 2022-10-24 20:27:36,295 - mmcv - INFO - Epoch(val) [3][313]      acc: 0.4235
(MMCVRunner pid=21452) 2022-10-24 20:28:18,232 - mmcv - INFO - Epoch(val) [4][313]      acc: 0.4261
```