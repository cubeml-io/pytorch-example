import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int)
parser.add_argument("--node_rank", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()


def example(local_rank, node_rank, local_size, world_size):
    # 初始化
    rank = local_rank + node_rank * local_size
    dist.init_process_group("gloo",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    # 创建模型
    model = nn.Linear(10, 10)
    # 放入DataParallel
    parallel_model = DataParallel(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(parallel_model.parameters(), lr=0.001)
    # 进行前向后向计算
    for i in range(100):
        if rank == 1:
            time.sleep(1)
        print("train step ", i)
        inputs = torch.randn(20, 10)
        outputs = parallel_model(inputs)
        labels = torch.randn(20, 10)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def main():
    local_size = 1
    print("local_size: %s" % local_size)
    mp.spawn(example,
             args=(args.node_rank, local_size, args.world_size,),
             nprocs=local_size,
             join=True)


if __name__ == "__main__":
    main()
# python ./code/pytorch_distributed_demo_cpu.py --world_size=2 --node_rank=${VC_TASK_INDEX} --master_addr=${VC_MASTER_HOSTS%%,*} --master_port=12345