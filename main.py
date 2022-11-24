# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import transformers
print(transformers.__version__)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import torch
    print(torch.__version__)
    torch_tril = torch.tril(torch.rand(2,3, 4))
    print(torch_tril)
    print(torch.masked_fill(torch_tril,torch_tril==0,-float("inf")).softmax(dim=-1))
    in_range_ = [torch.nn.Linear(5, 5) for _ in range(2)]
    nn_sequential = torch.nn.Sequential(*in_range_)
    print(nn_sequential)
    module_list = torch.nn.ModuleList(in_range_)
    print(module_list)
    rand = torch.rand(4, 5)
    sequential = nn_sequential(rand)
    list1 = module_list(rand)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
