

'''
dataset & dataloader
'''


'''
학습 관련 설정
'''
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
global_step = 0
first_epoch = 0

'''
resume code
- 어떤 체크포인트 가져올지 선택 : 최신을 가져올지
- step 재조정
'''
import os

if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = ...
    else:
        # output_dir에서 checkpoint로 시작하는 것 중에 뒷 부분 숫자가 높은 디렉토리 주소
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        # 발견되지 않으면 resume 안하는 걸로 처리
        args.resume_from_checkpoint = None
    else:
        # 체크포인트 저장되는 폴더에서 가져옴
        accelerator.load_state(os.path.join(args.output_dir, path))
        # 체크포인트 이름이 표시됨 (저장할 떄는 accumulation 반영 x)
        global_step = int(path.split("-")[1])

        # grad acc 고려해서 step 제대로 계산
        resume_global_step = global_step * args.gradient_accumulation_steps
        initial_global_step = global_step

else:
    initial_global_step = 0
'''
Training loop
- 대부분 epoch 기반인거 같으니 epoch 기반으로 간다
'''
from tqdm.auto import tqdm
progress_bar = tqdm(
    range(global_step, args.max_train_steps),
    initial=initial_global_step,
    disable=not accelerator.is_local_main_process
)
progress_bar.set_description('steps')

# epoch 기반
for epcoh in range(first_epoch, args.num_train_epochs):
    # epoch 초반 train mode
    model.train()
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        progress_bar.set_description("Global step: {}".format(global_step))
        # grad acc + flash attention
        with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(enable_flash=not args.disable_flashattention, enable_mem_efficient=not args.disable_flashattention, enable_math=True):
            return_dict = model(batch, noise_scheduler)
            loss = return_dict["loss"]
            
    