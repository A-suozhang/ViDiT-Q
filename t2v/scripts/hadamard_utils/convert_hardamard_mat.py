import torch

file_paths=["had.36.pal2.txt","had.36.will.txt","had.144.tpal.txt"]

d = {}

for file_path in file_paths:

    rows = []
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split the line into characters
            line = line.strip()
            row_values = []
            for char in line:
                if char == '+':
                    row_values.append(1)
                elif char == '-':
                    row_values.append(-1)
            rows.append(row_values)

    # Convert the list of rows to a 2-D PyTorch tensor
    tensor_2d = torch.tensor(rows, dtype=torch.float32)

    d[file_path.strip('.txt')] = tensor_2d

torch.save(d, 'hadamard_mat.pth')

