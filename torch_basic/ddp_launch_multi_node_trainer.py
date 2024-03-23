"""
PL, transformers 처럼 Trainer를 만들어서 사용

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

"""

class Trainer:
    def __init__(self, model, input, loss_fn, optimizer, scheduler, 
                 metric, epochs, train_loader, eval_loader):
        ...
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    # outputs이 [B, Num of class]라 가정
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if scaler:
                    loss = scaler.scale(loss)
                    loss.backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Batch 하나가 끝날 때마다 running metrics update
                # Batch size를 고려해서 넣음
                running_loss += loss.item() * inputs.size(0) # 배치 곫하는게 맞나?
                running_corrects += torch.sum(preds == labels.data)

            self.scheduler.step()
            #### Epoch metrics ############
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size


            self.evaluate()

    def evaluate(self):
        self.model.eval()
        correct = 0
        for data in self.eval_loader:
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred==data.y).sum())
            
        return correct / len(self.eval_loader.dataset)

    def predict(self, eval_loader):
        self.model.eval()
        