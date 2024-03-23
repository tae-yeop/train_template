"""
다수의 모델이 개별적인 optimizer로 학습할 때
다수의 모델이 개별적인 아웃풋을 내고 이 아웃풋을 합쳐서 학습하는 경우
"""

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        # (retain_graph here is unrelated to amp, it's present because in this
        # example, both backward() calls share some sections of graph.)
        # 여기선 loss0에 대해 backward를 하고 그래프를 유지해야한다 => 안그러면 loss1이 끊어짐?
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()


        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
        