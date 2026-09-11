import torch

def main():
    # Load the checkpoint
    checkpoint = torch.load('final.pt', map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    print("All keys in checkpoint:")
    for key in sorted(state_dict.keys()):
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else state_dict[key].size()
        print(f"  {key}: {shape}")




if __name__ == "__main__":
    main()
