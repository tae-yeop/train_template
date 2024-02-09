EMAModel









# 모든 학습 끝나고 랭크0에서
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    # 특정 모델만 저장한다면
    unet = accelerator.unwrap_model(unet)
    # EMA 사용했다면
    if args.use_ema:
        ema_unet.copy_to()