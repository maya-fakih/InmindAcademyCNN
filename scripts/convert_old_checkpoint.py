"""
One-time use: converts an old-format checkpoint (just a raw state_dict, from
before we added resume support) into the new format train.py expects.

Usage:
    uv run python scripts/convert_old_checkpoint.py weights/03-resnet-blocks.pth weights/03-resnet-blocks-best.pth 99 90.52

Args: <input_path> <output_path> <epoch_number> <best_val_acc>

Note: optimizer state can't be recovered (it was never saved), so momentum
buffers restart fresh. This causes a tiny, one-time hiccup in training
smoothness right after resuming, not a real problem — SGD momentum rebuilds
within a few batches.
"""
import sys
import torch

def main():
    if len(sys.argv) != 5:
        print("Usage: python convert_old_checkpoint.py <input_path> <output_path> <epoch> <best_val_acc>")
        sys.exit(1)

    input_path, output_path, epoch, best_val_acc = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])

    old_state_dict = torch.load(input_path, map_location='cpu')

    new_checkpoint = {
        'epoch': epoch,
        'model_state_dict': old_state_dict,
        'optimizer_state_dict': None,  # not recoverable, train.py will detect this and start optimizer fresh
        'best_val_acc': best_val_acc,
    }
    torch.save(new_checkpoint, output_path)
    print(f"Converted {input_path} -> {output_path} (epoch={epoch}, best_val_acc={best_val_acc})")

if __name__ == '__main__':
    main()