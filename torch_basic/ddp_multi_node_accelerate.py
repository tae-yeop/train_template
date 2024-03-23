from tqdm.auto import tqdm
progress_bar = tqdm(
    range(global_step, args.max_train_steps),
    initial=initial_global_step,
    disable=not accelerator.is_local_main_process
)
progress_bar.set_description('steps')