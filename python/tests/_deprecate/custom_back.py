import torch 
# Merge inside vs outside. 

class Shuffle(torch.autograd.Function):

    # from_sizes: Shapes exppected from other gpus.
    # To offsets that iwll be shuffled.
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b  = b
        return a * b
    
    @staticmethod
    def backward(ctx, grad):
        
        return 10 * torch.ones(ctx.a.shape), 2 * torch.ones(ctx.b.shape)

        
a = torch.ones(10,10, requires_grad = True)   
b = torch.ones(10,10, requires_grad = True)
c = Shuffle.apply(a,b)


(torch.sum(c) + torch.sum(a)).backward()

"Operations inside a cusotm layer do not trigger gradients"
"However returned gradients are approproately merged"

print(a.grad)

