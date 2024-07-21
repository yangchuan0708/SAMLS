import torch

net = torch.nn.Linear(4, 3)

input_t = torch.randn(4)

# with torch.no_grad():

for name, param in net.named_parameters():
    param.requires_grad = True
    print("{} {}".format(name, param.requires_grad))

out = net(input_t)

for name, param in net.named_parameters():

    print("{} {}".format(name, param.grad))

print('Output: {}'.format(out))

print('Output requires gradient: {}'.format(out.requires_grad))
print('Gradient function: {}'.format(out.grad))