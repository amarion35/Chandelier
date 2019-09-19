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

The framework also supports GAN training:
```
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(1, 32, (3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(64, 32, (3,3), padding=(1,1))
        self.fc1 = nn.Linear(2048,1)

    def forward(self, x, training):
        x = x.view(-1, 1, 8, 8)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.4, training=training)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.4, training=training)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.4, training=training)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, input_shape):
        super(Classifier, self).__init__()
        self.input_shape = input_shape
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
        
generator = Generator(input_shape=10)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999), eps=1e-8)
generator_loss = nn.BCELoss(reduction='mean')
generator_model = Model(generator, device='cuda:2')
generator_model.compile(optimizer=generator_optimizer, loss=generator_loss)

discriminator = Discriminator(input_shape=64)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999), eps=1e-8)
discriminator_loss = nn.BCELoss(reduction='mean')
discriminator_metrics = [binary_accuracy]
discriminator_model = Model(discriminator, device='cuda:2')
discriminator_model.compile(optimizer=discriminator_optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

gan_loss = nn.BCELoss(reduction='mean')
gan_metrics = [binary_accuracy]
gan = GAN(discriminator_model, generator_model, loss=gan_loss, metrics=gan_metrics, device='cuda:2')
gan.fit(X_train, batch_size=64, epochs=200)

plt.figure()
plt.plot(gan.hist['d_loss'], label='d_loss')
plt.plot(gan.hist['val_d_loss'], label='val_d_loss')
plt.plot(gan.hist['g_loss'], label='g_loss')
plt.plot(gan.hist['val_g_loss'], label='val_g_loss')
plt.legend()
plt.savefig('loss')

# keys signification
# real : performances on real data
# fake : performances on fake data
# d : discriminator
# g : generator
# val : validation
for metric in discriminator_metrics:
    plt.figure()
    plt.plot(gan.hist['real_d_'+metric.__name__], label='real_d_'+metric.__name__)
    plt.plot(gan.hist['val_real_d_'+metric.__name__], label='val_real_d_'+metric.__name__)
    plt.plot(gan.hist['fake_d_'+metric.__name__], label='fake_d_'+metric.__name__)
    plt.plot(gan.hist['val_fake_d_'+metric.__name__], label='val_fake_d_'+metric.__name__)
    plt.plot(gan.hist['d_'+metric.__name__], label='d_'+metric.__name__)
    plt.plot(gan.hist['val_d_'+metric.__name__], label='val_d_'+metric.__name__)
    plt.plot(gan.hist['g_'+metric.__name__], label='g_'+metric.__name__)
    plt.plot(gan.hist['val_g_'+metric.__name__], label='val_g_'+metric.__name__)
    plt.legend()
    plt.savefig(metric.__name__)
        
```
