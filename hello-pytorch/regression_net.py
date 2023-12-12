import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        x_hidden = F.relu(self.hidden(x_input))
        x_predict = self.predict(x_hidden)
        return x_predict


if __name__ == '__main__':
    # fake data
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    # net
    net = Net(n_feature=1, n_hidden=10, n_output=1)
    # print(net)
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    # train
    plt.ion()
    for step in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 14, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    print("ok")
