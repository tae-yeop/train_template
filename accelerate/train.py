"""
https://github.dev/huggingface/diffusers/blob/main/examples/consistency_distillation/train_lcm_distill_lora_sdxl.py
https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py
"""

'''
dataset & dataloader
'''

import torch
from accelerate import Accelerator



# 유틸
# torch.compile()이 되는 경우 고려해서
def unwrap_model(model):
    from diffusers.utils.torch_utils import is_compiled_module
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


'''
학습 관련 설정
'''
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
global_step = 0
first_epoch = 0


optimizer = ...
accelerator = Accelerator(gradient_accumulation_steps=..., # grad_acc
                          )


model, dataloader = accelerator.prepare(model, dataloader)




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
            # Mixed precision
            with accelerator.autocast():
                return_dict = model(batch, noise_scheduler)
                loss = return_dict["loss"]
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params, arg.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)


        if accelerator.sync_gradients:
            # 저장할 때 global_step 기준
            if accelerator.is_local_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)


# 모든 학습 끝나고 랭크0에서
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    # 특정 모델만 저장한다면
    model = accelerator.unwrap_model(model)

    # 허브 업로드
    if args.push_to_hub:
        