from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import hstack
from torch.nn import ReLU, Sigmoid, Linear, Sequential, LogSoftmax
from torch.nn.functional import binary_cross_entropy, kl_div, mse_loss
from torch.optim import Adam


class JustPolicyNet(LightningModule):
    def __init__(self):
        super().__init__()
        l1 = Linear(4096, 2000)
        l2 = Linear(2000, 2272)

        act = ReLU(inplace=True)

        self.body = Sequential(l1, act, l2)
        self.policy_head = Sigmoid()

    def forward(self, x):
        return self.policy_head(self.body(x))

    def predict(self, x):
        boo = self.forward(x) > 0.5
        return boo.float()

    def training_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('loss', loss)
        self.log('ba', ba, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_ba', ba, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_ba', ba)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _loss(self, batch, batch_idx):
        x, y = batch
        y_pred = self.policy_head(self.body(x))
        loss = binary_cross_entropy(y_pred, y)

        y_pred_ = (y_pred > 0.5).float()
        ba = accuracy(y_pred_, y, class_reduction='macro')
        return loss, ba


class DoubleHeadedNet(JustPolicyNet):
    def __init__(self):
        super().__init__()

        self.policy_net = JustPolicyNet.load_from_checkpoint('net.ckpt')

        self.body = self.policy_net.body

        self.policy_head = LogSoftmax()

        self.value_head = Sequential(
            Linear(2273, 1),
        )

    def forward(self, x):
        x_policy, x_value = x[:, :-1], x[:, -1]
        body = self.body(x_policy)

        policy = self.policy_head(body)
        stack = hstack((body, x_value))

        value = self.value_head(stack)
        return policy, value

    def predict(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)
        opt = self.optimizers()

        self.manual_backward(loss_policy, opt, retain_graph=True)
        self.manual_backward(loss_value, opt)

        self.log('loss_policy', loss_policy)
        self.log('loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)

        self.log('val_loss_policy', loss_policy)
        self.log('val_loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)

        self.log('test_loss_policy', loss_policy)
        self.log('test_loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-4)
        return optimizer

    def _losses(self, batch, batch_idx):
        x, y = batch

        finger, depth = x[:, :-1], x[:, -1].reshape(-1, 1)
        y_policy, y_value = y[:, :-1], y[:, -1].reshape(-1, 1)

        pred_policy = self.policy_net(finger)
        stack = hstack((self.body(finger), depth))

        pred_value = self.value_head(stack)

        loss_policy = kl_div(pred_policy, y_policy, reduction='batchmean')
        loss_value = mse_loss(pred_value, y_value)

        return loss_policy, loss_value
