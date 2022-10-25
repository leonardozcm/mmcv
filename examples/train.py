import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from mmcv import runner

from mmcv.parallel import MMDataParallel
from mmcv.runner import EpochBasedRunner
from mmcv.runner import EvalHook
from mmcv.utils import get_logger


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, data, optimizer):
        images, labels = data
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        return {'loss': loss}

    def val_step(self, data, optimizer):
        inputs, targets = data
        correct = 0.0
        total = 0
        with torch.no_grad():
            preds = torch.argmax(self(inputs), dim=-1)
            correct = torch.sum(torch.eq(preds, targets)).item()
            total = targets.numel()

        return {"correct": correct, "total": total}


from bigdl.orca.learn.ray_estimator import Estimator as OrcaRayEstimator
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch.pytorch_ray_estimator import get_driver_node_ip, check_for_failure
from bigdl.orca.learn.pytorch.utils import find_free_port
from bigdl.orca.learn.pytorch.torch_runner import TorchDistBackend

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import mmcv
from mmcv.parallel import MMDataParallel

from typing import (Any, Dict, List, Optional, Tuple, Callable, overload)
import ray


class MMCVRunner(EpochBasedRunner):
    EBR_slots = (
        "model",
        "batch_processor",
        "optimizer",
        "logger",
        "meta",
        "work_dir",
        "_model_name",
        "_rank",
        "_world_size",
        "timestamp",
        "mode",
        "_hooks",
        "_epoch",
        "_iter",
        "_inner_iter",
        "_max_epochs",
        "_max_iters",
        "log_buffer",
    )  # Here just for declaration

    def __init__(self, runner_creator=None, config=None) -> None:
        self.runner_creator = runner_creator
        self.config = config
        self.eval_result = {}

    # Expose attrs of EBR here for concise when overriding `train_step` etc. in EpochBasedRunner,
    # for example:
    #    before: self.runnner.model(batch)
    #    now: self.model(batch)
    def _wrapFromEBR(self, ebrunner):
        for k in self.EBR_slots:
            # todo: check neccessary components
            setattr(self, k, getattr(ebrunner, k))

    def get_eval_result(self):
        return self.eval_result

    def setup(self, cores_per_node):
        import torch
        torch.set_num_threads(cores_per_node)

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
        self.model = MMDistributedDataParallel(self.model)  # runner.model: `torch.nn.Module`

    def setup_components(self):
        runner = self.runner_creator(self.config)

        self._wrapFromEBR(runner)

    def with_sampler(self, loader):
        data_loader_args = {
            "dataset": loader.dataset,
            "batch_size": loader.batch_size,
            "shuffle": False,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "sampler": DistributedSampler(loader.dataset,
                                          num_replicas=self.size,
                                          rank=self.world_rank)
        }
        return DataLoader(**data_loader_args)

    def train(self, data_loader, **kwargs):
        return super().train(data_loader, **kwargs)

    def run(self,
            data_loaders_creators: List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,  # deprecated
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        data_loaders = [self.with_sampler(dataloader_creator(self.config)) for dataloader_creator in
                        data_loaders_creators]
        super().run(data_loaders, workflow, max_epochs, **kwargs)


class MMCVEstimator(OrcaRayEstimator):
    def __init__(self, runner_creator=None, config=None) -> None:
        super().__init__()
        self.runner_creator = runner_creator
        ray_ctx = OrcaRayContext.get()

        workers_per_node = 1
        cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
        num_nodes = ray_ctx.num_ray_nodes * workers_per_node
        RemoteRunner = ray.remote(num_cpus=cores_per_node)(MMCVRunner)

        params = dict(runner_creator=self.runner_creator,
                      config=config)

        self.remote_workers = [
            RemoteRunner.remote(**params) for _ in range(num_nodes)
        ]
        ray.get([
            worker.setup.remote(cores_per_node)
            for i, worker in enumerate(self.remote_workers)
        ])

        driver_ip = get_driver_node_ip()
        driver_tcp_store_port = find_free_port()

        _ = dist.TCPStore(driver_ip, driver_tcp_store_port, -1, True,
                          dist.constants.default_pg_timeout)

        ray.get([
            worker.setup_torch_distribute.remote(
                driver_ip, driver_tcp_store_port, i, num_nodes)
            for i, worker in enumerate(self.remote_workers)
        ])

    def fit(self,
            data_loaders: List[Callable],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,  # deprecated
            **kwargs) -> None:
        params = dict(data_loaders_creators=data_loaders,
                      workflow=workflow,
                      max_epochs=max_epochs,
                      **kwargs)
        success, worker_stats = self._train_epochs(**params)
        print(worker_stats)

        eval_result_from_workers = ray.get([worker.get_eval_result.remote()
                                            for worker in self.remote_workers])
        self._print_eval_results(eval_result_from_workers)

        epoch_stats = list(map(list, zip(*worker_stats)))
        for i in range(len(epoch_stats)):
            epoch_stats[i] = self._process_stats(epoch_stats[i])
        return epoch_stats

    def _print_eval_results(self, eval_results):
        keys = eval_results[0]

        print_dict = {}

        for k in keys:
            correct = 0.0
            total = 0
            for d in eval_results:
                cur_list = d[k]
                for c, t in cur_list:
                    correct += c
                    total += t
            print_dict[k] = correct / total
        print(print_dict)


    def _train_epochs(self, **params) -> None:
        remote_worker_stats = []
        for i, w in enumerate(self.remote_workers):
            stats = w.run.remote(**params)
            remote_worker_stats.append(stats)

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, ray.get(remote_worker_stats)
        else:
            return success, None

    def _process_stats(self, worker_stats):
        # TODO: worker_stats returns None for mmcv runner do so
        # Need to sync worker_stats in driver.
        print(worker_stats)

        return worker_stats

    def evaluate(self, data, batch_size, num_steps=None):
        return super().evaluate(data, batch_size, num_steps)

    def predict(self, data, batch_size):
        return super().predict(data, batch_size)

    def get_model(self):
        return super().get_model()

    def save(self, model_path):
        return super().save(model_path)

    def load(self, model_path):
        return super().load(model_path)

    def shutdown(self):
        return super().shutdown()


if __name__ == '__main__':
    init_orca_context(cores=8, memory="8g")
    batch_size = 32


    def runner_creator(config):
        model = Model()
        if torch.cuda.is_available():
            # only use gpu:0 to train
            # Solved issue https://github.com/open-mmlab/mmcv/issues/1470
            model = MMDataParallel(model.cuda(), device_ids=[0])

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

        # Evaluation Part(Optional)
        from typing import OrderedDict
        def evaluate(self, results, logger=None):
            acc = results.item()
            output = OrderedDict(
                acc=acc
            )
            return output

        def cifar10_test_fn(model: nn.Module, data_loader: DataLoader) -> list:
            model.eval()
            results = []
            dataset = data_loader.dataset
            prog_bar = mmcv.ProgressBar(len(dataset))

            correct = torch.tensor(0)
            total = torch.tensor(0)

            for inputs, targets in data_loader:
                with torch.no_grad():
                    preds = torch.argmax(model(inputs), dim=-1)
                    correct += torch.sum(torch.eq(preds, targets))
                    total += targets.numel()
                batch_size = len(inputs)
                for _ in range(batch_size):
                    prog_bar.update()

            return correct.float() / total

        from types import MethodType
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valset = CIFAR10(
            root='data', train=False, download=True, transform=transform)
        valset.evaluate = MethodType(evaluate, valset)

        valloader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=2)
        eval_hook = DistEvalHook(valloader, interval=1, test_fn=cifar10_test_fn, greater_keys=["acc"])
        runner.register_hook(eval_hook)

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


    def val_dataloader_creator(config):
        # dataset and dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valset = CIFAR10(
            root='data', train=False, download=True, transform=transform)

        valloader = DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=2)
        return valloader


    est = MMCVEstimator(runner_creator=runner_creator, config={})
    est.fit([train_dataloader_creator, val_dataloader_creator], [('train', 1), ('val', 1)])

    stop_orca_context()


from collections import defaultdict

class DistEvalHook(EvalHook):
    def __init__(self,
                 dataloader: DataLoader,
                 start: Optional[int] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 save_best: Optional[str] = None,
                 rule: Optional[str] = None,
                 test_fn: Optional[Callable] = None,
                 greater_keys: Optional[List[str]] = None,
                 less_keys: Optional[List[str]] = None,
                 out_dir: Optional[str] = None,
                 file_client_args: Optional[dict] = None,
                 **eval_kwargs):
        if test_fn is None:
            from mmcv.engine import multi_gpu_test
            test_fn = multi_gpu_test

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            **eval_kwargs)

    def before_run(self, runner):
        runner.eval_result = defaultdict(list)

    def after_val_iter(self, runner):
        runner.eval_result[runner._epoch].append((runner.outputs['correct'], runner.outputs['total']))
