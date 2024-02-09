
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--local-rank', type=int, help='local rank, will passed by ddp')
args = parser.parse_args()

if args.local_rank is None and 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])

# 장치 설정
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda')
# device = torch.device('cuda', args.local_rank) : 이렇게 해도 된다


dist.init_process_group(backend='nccl', init_method='env://')
torch.manual_seed(args.seed)

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}  
cuda_kwargs = {'num_workers': 1, 'pin_memory': True,}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

# 먼저 rank0에서 다운로드
if args.local_rank == 0:
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
# 다운 완료될 떄까지 기다림
torch.distributed.barrier()

if args.local_rank > 0:
    dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)


train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1, shuffle=True)
test_sampler = torch.utils.data.SequentialSampler(dataset2)

train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, shuffle=False, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, sampler=test_sampler, **test_kwargs)

model = Net().to(device)
optimizer