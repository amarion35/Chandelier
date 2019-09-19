# chandelier

In a nutshell:
```
# Define your PyTorch model
class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape # <-- must contain the input_shape attribute
        self.fc1 = nn.Linear(input_shape,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)

    def forward(self, x, training):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

classifier = Classifier(input_shape=64)

classif_model = chandelier.Model(classifier, device='cuda:2') # <-- Model will manage training

optimizer = optim.Adam(classifier.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-8)
loss = nn.CrossEntropyLoss(reduction='mean')
metrics = [sparse_categorical_accuracy]
classif_model.compile(optimizer, loss, metrics=metrics) # <-- Define your optimizer, loss and metrics

classif_model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_data=(X_test, Y_test)) # <-- fit your model

# Plot losses
plt.figure()
plt.plot(classif_model.hist['loss'], label='loss')
plt.plot(classif_model.hist['val_loss'], label='val_loss')
plt.legend()
plt.savefig('loss')

# Plot your metrics
for metric in metrics:
    plt.figure()
    plt.plot(classif_model.hist[metric.__name__], label=metric.__name__)
    plt.plot(classif_model.hist['val_'+metric.__name__], label='val_'+metric.__name__)
    plt.legend()
    plt.savefig(metric.__name__)
```
