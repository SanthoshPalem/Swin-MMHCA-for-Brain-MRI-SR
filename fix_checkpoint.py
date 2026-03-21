import torch

def convert_checkpoint(old_path, new_path):
    print(f"Loading old checkpoint: {old_path}")
    checkpoint = torch.load(old_path, map_location='cpu')
    
    # If the checkpoint is a dict with 'model_state_dict', extract it
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # --- 1. Path SR Mapping ---
        if key.startswith("cnn_encoders."):
            new_key = "path_sr." + key
        elif key.startswith("pos_encoder."):
            new_key = "path_sr." + key
        elif key.startswith("transformer_encoder."):
            new_key = key.replace("transformer_encoder.", "path_sr.transformer.")
            
        # --- 2. Decoder Mapping ---
        elif key.startswith("upsample_block1.upsample."):
            new_key = key.replace("upsample_block1.upsample.", "sr_decoder.0.upsample.")
        elif key.startswith("upsample_block2.upsample."):
            new_key = key.replace("upsample_block2.upsample.", "sr_decoder.1.upsample.")
        elif key.startswith("upsample_block3.upsample."):
            new_key = key.replace("upsample_block3.upsample.", "sr_decoder.2.upsample.")
        elif key.startswith("upsample_block4.upsample."):
            new_key = key.replace("upsample_block4.upsample.", "sr_decoder.3.upsample.")
        elif key.startswith("upsample_block5.upsample."):
            new_key = key.replace("upsample_block5.upsample.", "sr_decoder.4.upsample.")
        elif key.startswith("upsample_block6.upsample."):
            new_key = key.replace("upsample_block6.upsample.", "sr_decoder.5.upsample.")
        elif key.startswith("final_conv."):
            new_key = key.replace("final_conv.", "sr_decoder.6.")
            
        # These modules from the old architecture are discarded/replaced in the new one:
        # mmhca, skip_conv, conv_concat_skip
        if not key.startswith(("mmhca.", "skip_conv.", "conv_concat_skip.")):
             new_state_dict[new_key] = value

    # Save the mapped state_dict
    torch.save(new_state_dict, new_path)
    print(f"Successfully saved mapped checkpoint to {new_path}")

if __name__ == '__main__':
    convert_checkpoint(
        "epoch_checkpoints/swin_mmhca_x4_epoch_400.pth", 
        "epoch_checkpoints/swin_mmhca_x4_epoch_400_fixed.pth"
    )
