import json
import os
from typing import Dict, Any

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    WanPipeline,
    WanImageToVideoPipeline,
    UniPCMultistepScheduler
)
from transformers import AutoTokenizer, UMT5EncoderModel

from model_network.turbo_wan import WanTransformer3DModel


def get_transformer_config(model_type: str) -> Dict[str, Any]:
    if model_type == "Wan-T2V-1.3B":
        return {
            "added_kv_proj_dim": None,
            "attention_head_dim": 128,
            "cross_attn_norm": True,
            "eps": 1e-06,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "in_channels": 16,
            "num_attention_heads": 12,
            "num_layers": 30,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "qk_norm": "rms_norm_across_heads",
            "text_dim": 4096,
            "attention_type": "sla",
            "sla_topk": 0.1
        }
    elif model_type == "Wan-T2V-14B":
        return {
            "added_kv_proj_dim": None,
            "attention_head_dim": 128,
            "cross_attn_norm": True,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_channels": 16,
            "num_attention_heads": 40,
            "num_layers": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "qk_norm": "rms_norm_across_heads",
            "text_dim": 4096,
            "attention_type": "sla",
            "sla_topk": 0.1
        }
    elif model_type == "Wan2.2-I2V-14B-720p":
        return {
            "added_kv_proj_dim": None,
            "attention_head_dim": 128,
            "cross_attn_norm": True,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_channels": 36,
            "num_attention_heads": 40,
            "num_layers": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "qk_norm": "rms_norm_across_heads",
            "text_dim": 4096,
            "attention_type": "sla",
            "sla_topk": 0.1
        }
    else:
        raise NotImplementedError("unsupported model type")


def modify_model_index_json(model_type,output_dir):
    model_index_path = os.path.join(output_dir, "model_index.json")

    with open(model_index_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # change class name
    if "_class_name" in config:
        if model_type == "Wan2.2-I2V-14B-720p":
            config["_class_name"] = "WanImageToVideoDmdPipeline"
        else:
            config["_class_name"] = "WanDMDPipeline"
    else:
        raise KeyError("_class_name not found in model_index.json")

    # change transformer
    if "transformer" in config:
        if isinstance(config["transformer"], list) and len(config["transformer"]) == 2:
            config["transformer"][0] = "diffusers"
        else:
            raise ValueError("Unexpected format for 'transformer' in model_index.json")
    else:
        raise KeyError("'transformer' not found in model_index.json")

    if model_type == "Wan2.2-I2V-14B-720p" and "transformer_2" in config:
        if isinstance(config["transformer_2"], list) and len(config["transformer_2"]) == 2 and config["transformer_2"][0]:
            config["transformer_2"][0] = "diffusers"
        else:
            raise ValueError("Unexpected format for 'transformer' in model_index.json")


    # save
    with open(model_index_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Modified model_index.json: _class_name → WanDMDPipeline, transformer[0] → diffusers")


def convert_transformer_from_pth(model_type, pth_path):
    """Transformer"""
    # Load the original weight
    original_state_dict = torch.load(pth_path, map_location='cpu')
    if 'patch_embedding.weight' in original_state_dict:
        weight = original_state_dict['patch_embedding.weight']
        if weight.dim() == 2:  # if [1536, 64]
            # [1536, 64] --> [1536, 16, 1, 2, 2]
            # [5120, 64] --> [5120, 16, 1, 2, 2]
            # 64 = 16 * 1 * 2 * 2
            if model_type in ("Wan-T2V-14B","Wan-T2V-1.3B"):
                original_state_dict['patch_embedding.weight'] = weight.view(-1, 16, 1, 2, 2)
            elif model_type in ("Wan2.2-I2V-14B-720p"):
                original_state_dict['patch_embedding.weight'] = weight.view(-1, 36, 1, 2, 2)
    # Creating a Wan Transformer Configuration
    config = get_transformer_config(model_type)
    sample_tensor = next(iter(original_state_dict.values()))
    # Create a transformer model
    transformer = WanTransformer3DModel(**config)
    # print("torch_dtype=sample_tensor.dtype=",sample_tensor.dtype)

    # Weight mapping (adjusts to .pth file structure)
    key_mapping = {
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj",
        "head.modulation": "scale_shift_table",
        "head.head": "proj_out",
        "modulation": "scale_shift_table",
        "ffn.0": "ffn.net.0.proj",
        "ffn.2": "ffn.net.2",
        # Hack to swap the layer names
        # The original model calls the norms in following order: norm1, norm3, norm2
        # We convert it to: norm1, norm2, norm3
        "norm2": "norm__placeholder",
        "norm3": "norm2",
        "norm__placeholder": "norm3",
        # For the I2V model
        "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
        "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
        "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
        "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
        # for the FLF2V model
        "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
        # Add attention component mappings
        "self_attn.q": "attn1.to_q",
        "self_attn.k": "attn1.to_k",
        "self_attn.v": "attn1.to_v",
        "self_attn.o": "attn1.to_out.0",
        "self_attn.attn_op.local_attn.proj_l": "attn1.attn_op.local_attn.proj_l",
        "self_attn.norm_q": "attn1.norm_q",
        "self_attn.norm_k": "attn1.norm_k",
        "cross_attn.q": "attn2.to_q",
        "cross_attn.k": "attn2.to_k",
        "cross_attn.v": "attn2.to_v",
        "cross_attn.o": "attn2.to_out.0",
        "cross_attn.norm_q": "attn2.norm_q",
        "cross_attn.norm_k": "attn2.norm_k",
        "attn2.to_k_img": "attn2.add_k_proj",
        "attn2.to_v_img": "attn2.add_v_proj",
        "attn2.norm_k_img": "attn2.norm_added_k",
    }

    # Convert weights
    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in key_mapping.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    # Load the transformed weights
    transformer.load_state_dict(original_state_dict, strict=True, assign=True)

    return transformer


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def convert_vae_from_pth(vae_pth_path, output_dir):
    """Convert VAE section"""
    old_state_dict = torch.load(vae_pth_path, weights_only=True)
    new_state_dict = {}

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in old_state_dict.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            new_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            new_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            new_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            new_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Convert to down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Convert residual block naming but keep the original structure
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")

            new_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Convert to up_blocks
            parts = key.split(".")
            block_idx = int(parts[2])

            # Group residual blocks
            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx = 0
                    resnet_idx = block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx = 1
                    resnet_idx = block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx = 2
                    resnet_idx = block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx = 3
                    resnet_idx = block_idx - 12
                else:
                    # Keep as is for other blocks
                    new_state_dict[key] = value
                    continue

                # Convert residual block naming
                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key

                new_state_dict[new_key] = value

            # Handle shortcut connections
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")

                new_state_dict[new_key] = value

            # Handle upsamplers
            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

                new_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan()
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


def create_diffusers_pipeline(model_dir, transformer,transformer2, vae, umt5_pth, tokenizer_path):
    """Create a complete diffusers pipeline"""

    # text_encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        umt5_pth,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # scheduler
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=3.0
    )

    # pipeline
    if transformer2 is not None:
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            transformer_2=transformer2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.9,
        )
    else:
        pipe = WanPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    # save
    pipe.save_pretrained(model_dir, safe_serialization=True)

    return pipe


# main
def main():
    model_type = "Wan-T2V-1.3B"
    transformer_pth = "Wan2.1-T2V-1.3B/TurboWan2.1-T2V-1.3B-480P.pth"
    transformer2_pth = None
    vae_pth = "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    tokenizer_path = "google/umt5-xxl"
    umt5_pth = "google/umt5-xxl"
    transformer2 = None
    output_dir = "TurboWan2.1-T2V-1.3B-Diffusers"

    os.makedirs(output_dir, exist_ok=True)

    print("convert transformer...")
    transformer = convert_transformer_from_pth(model_type, transformer_pth)
    if model_type=="Wan2.2-I2V-14B-720p" and transformer2_pth:
        transformer2 = convert_transformer_from_pth(model_type, transformer2_pth)
    print("convert VAE...")
    vae = convert_vae_from_pth(vae_pth, output_dir)

    print("convert pipeline...")
    pipe = create_diffusers_pipeline(output_dir, transformer,transformer2 , vae, umt5_pth, tokenizer_path)

    print(f"The conversion is complete! Models are saved at: {output_dir}")

    print("modify model_index.json...")
    modify_model_index_json(model_type,output_dir)


if __name__ == "__main__":
    main()
