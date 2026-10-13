import torch


def hor_t(tensor, move_tensor):
    return (tensor + move_tensor)

def rot(tensor, x, y, z):
  rot_mat = torch.stack([torch.cos(z)*torch.cos(y), -torch.sin(x)*torch.sin(y)*torch.cos(z) - torch.cos(x)*torch.sin(z), -torch.cos(z)*torch.cos(x)*torch.sin(y)+torch.sin(z)*torch.sin(x),
                         torch.cos(y)*torch.sin(z), -torch.sin(z)*torch.sin(y)*torch.sin(x)+torch.cos(z)*torch.cos(x), -torch.cos(x)*torch.sin(y)*torch.sin(z)-torch.cos(z)*torch.sin(x),
                         torch.sin(y), torch.sin(x)*torch.cos(y), torch.cos(y)*torch.cos(x)]).view(3,3)
  return torch.transpose(torch.matmul(rot_mat, torch.transpose(tensor, 0, 1)),  0, 1)

def rotate_and_move(input_tensor, vs):
    return rot(hor_t(input_tensor, vs[:3]), vs[3], vs[4], vs[5])