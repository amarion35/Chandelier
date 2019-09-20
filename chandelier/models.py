import numpy as np
import torch

class Model:
    def __init__(self, model, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model.to(self.device)
        self.hist = {
            'loss': [],
            'val_loss': []
        }

    def compile(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.loss_function = loss
        self.metrics = metrics
        for metric in metrics:
            self.hist[metric.__name__] = []
            self.hist['val_'+metric.__name__] = []
    
    def fit(self, x, y, batch_size, epochs=1, verbose=1, validation_data=None, initial_epoch=0):
        def vprint(*args, **kwargs):
            if verbose==1:
                print(*args, **kwargs)
        assert len(x)==len(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if not validation_data is None:
            val_x = validation_data[0]
            val_y = validation_data[1]
            assert len(val_x)==len(val_y)
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)
        idx = np.arange(len(x))
        n_batches = int(np.ceil(len(x)/batch_size))
        for i in np.arange(epochs)+initial_epoch:
            vprint('Epoch {}'.format(i), end='')
            np.random.shuffle(idx)
            for b in range(n_batches):
                batch = idx[batch_size*b:batch_size*(b+1)]
                x_batch = x[batch]
                y_batch = y[batch]

                self.optimizer.zero_grad()
                out = self.model(x_batch, training=True)
                loss = self.loss_function(out, y_batch)
                loss.backward(retain_graph=True)
                self.optimizer.step()
            metrics = self.eval(x, y, None)
            self._record(metrics)
            vprint(' - loss: {:.4f}'.format(self.hist['loss'][-1]), end='')
            if False:
                vprint(' - val_loss: {:.4f}'.format(self.hist['val_loss'][-1]), end='')
            vprint('')

    def eval(self, x, y, validation_data):
        out = self.model(x, training=False)
        loss = self.loss_function(out, y)
        
        results = {}

        results['loss'] = loss.cpu().data.numpy()
        for metric in self.metrics:
            results[metric.__name__] = metric(y, out)

        if not validation_data is None:
            val_x = validation_data[0]
            val_y = validation_data[1]
            out = self.model(val_x, training=False)
            loss = self.loss_function(out, val_y)
            results['val_loss'] = loss.cpu().data.numpy()
            for metric in self.metrics:
                results['val_'+metric.__name__] = metric(val_y, out)

        return results

    def _record(self, metrics):
        for k, v in metrics.items():
            self.hist[k].append(v)

    def predict(self, x, training=False):
        return self.model(x.to(self.device), training=False)


class GAN:
    def __init__(self, discriminator, generator, loss, metrics, device='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.discriminator = discriminator
        self.generator = generator
        self.loss_function = loss
        self.metrics = metrics
        self.hist = {
            'd_loss': [],
            'val_d_loss': [],
            'real_d_loss': [],
            'val_real_d_loss': [],
            'fake_d_loss': [],
            'val_fake_d_loss': [],
            'g_loss': [],
            'val_g_loss': []
        }
        for metric in metrics:
            self.hist['real_d_'+metric.__name__] = []
            self.hist['val_real_d_'+metric.__name__] = []
            self.hist['fake_d_'+metric.__name__] = []
            self.hist['val_fake_d_'+metric.__name__] = []
            self.hist['d_'+metric.__name__] = []
            self.hist['val_d_'+metric.__name__] = []
            self.hist['g_'+metric.__name__] = []
            self.hist['val_g_'+metric.__name__] = []

    def fit(self, x, batch_size, epochs=1, verbose=1, validation_data=None, initial_epoch=0):
        def vprint(*args, **kwargs):
            if verbose==1:
                print(*args, **kwargs)

        x = x.to(self.device)
        if not validation_data is None:
            val_x = validation_data
            val_x = val_x.to(self.device)

        idx = np.arange(len(x))
        n_batches = int(np.ceil(len(x)/batch_size))
        for i in np.arange(epochs)+initial_epoch:
            vprint('Epoch {}'.format(i), end='')
            np.random.shuffle(idx)
            for b in range(n_batches):
                batch = idx[batch_size*b:batch_size*(b+1)]
                real_x_batch = x[batch]
                real_y_batch = torch.empty(real_x_batch.size(0)*2, dtype=torch.float, device=self.device).fill_(1)
                noise = torch.empty(real_x_batch.size(0)*2, self.generator.model.input_shape, dtype=torch.float, device=self.device).normal_(0,1)
                fake_x_batch = self.generator.predict(noise, training=True)
                fake_y_batch = torch.empty(noise.size(0), dtype=torch.float, device=self.device).fill_(0)

                x_batch = torch.cat((real_x_batch, fake_x_batch[::2]))
                y_batch = torch.cat((real_y_batch[::2], fake_y_batch[::2]))

                # Discriminator
                self.discriminator.fit(x_batch, y_batch, batch_size=batch_size, epochs=1, verbose=0, initial_epoch=i)

                # Generator
                self.generator.optimizer.zero_grad()
                out = self.discriminator.predict(fake_x_batch)
                loss = self.loss_function(out, real_y_batch)
                loss.backward(retain_graph=True)
                self.generator.optimizer.step()
            metrics = self.eval(x, None)
            self._record(metrics)
            vprint(' - d_loss: {:.4f}'.format(self.hist['d_loss'][-1]), end='')
            vprint(' - g_loss: {:.4f}'.format(self.hist['g_loss'][-1]), end='')
            if False:
                vprint(' - val_loss: {:.4f}'.format(self.hist['val_loss'][-1]), end='')
            vprint('')

    def eval(self, x, validation_data):
        real_x = x
        real_y = torch.empty(real_x.size(0), dtype=torch.float, device=self.device).fill_(1)
        noise = torch.empty(real_x.size(0), self.generator.model.input_shape, dtype=torch.float, device=self.device).normal_(0,1)
        fake_x = self.generator.predict(noise, training=True)
        fake_y = torch.empty(real_x.size(0), dtype=torch.float, device=self.device).fill_(0)

        results = {}

        # Real data
        out = self.discriminator.predict(real_x)
        loss = self.loss_function(out, real_y)
        results['real_d_loss'] = loss.cpu().data.numpy()
        for metric in self.metrics:
            results['real_d_'+metric.__name__] = metric(real_y, out)

        # Fake data
        out = self.discriminator.predict(fake_x)
        loss = self.loss_function(out, fake_y)
        results['fake_d_loss'] = loss.cpu().data.numpy()
        for metric in self.metrics:
            results['fake_d_'+metric.__name__] = metric(fake_y, out)

        # All data
        out = self.discriminator.predict(torch.cat((real_x, fake_x)))
        loss = self.loss_function(out, torch.cat((real_y, fake_y)))
        results['d_loss'] = loss.cpu().data.numpy()
        for metric in self.metrics:
            results['d_'+metric.__name__] = metric(torch.cat((real_y, fake_y)), out)

        # Generator
        out = self.discriminator.predict(fake_x)
        loss = self.loss_function(out, real_y)
        results['g_loss'] = loss.cpu().data.numpy()
        for metric in self.metrics:
            results['g_'+metric.__name__] = metric(real_y, out)

        return results

    def _record(self, metrics):
        for k, v in metrics.items():
            self.hist[k].append(v)


    