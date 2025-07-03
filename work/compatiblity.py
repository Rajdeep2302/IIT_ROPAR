import torch
import os
from segment_anything import sam_model_registry
import segment_anything

def check_sam_compatibility(checkpoint_path):
    """Complete SAM model compatibility checker"""
    
    print("=" * 60)
    print("SAM MODEL COMPATIBILITY CHECKER")
    print("=" * 60)
    
    # 1. Check package versions
    print("\n1. PACKAGE VERSIONS")
    print("-" * 30)
    try:
        print(f"segment-anything version: {segment_anything.__version__}")
    except:
        print("segment-anything version: Unknown")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Available models: {list(sam_model_registry.keys())}")
    
    # 2. Check file info
    print("\n2. FILE INFORMATION")
    print("-" * 30)
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint file not found at {checkpoint_path}")
        return
    
    file_size = os.path.getsize(checkpoint_path) / (1024*1024)
    print(f"✅ File exists: {checkpoint_path}")
    print(f"File size: {file_size:.1f} MB")
    
    # 3. Expected sizes
    expected_sizes = {
        "vit_b": 358,
        "vit_l": 1249,
        "vit_h": 2564
    }
    
    print("\n3. SIZE COMPARISON")
    print("-" * 30)
    best_match = None
    min_diff = float('inf')
    
    for model_type, expected_size in expected_sizes.items():
        diff = abs(file_size - expected_size)
        status = "✅" if diff < 10 else "❌"
        print(f"{status} {model_type}: Expected {expected_size} MB, diff: {diff:.1f} MB")
        
        if diff < min_diff:
            min_diff = diff
            best_match = model_type
    
    print(f"\nBest size match: {best_match} (difference: {min_diff:.1f} MB)")
    
    # 4. Load and inspect checkpoint
    print("\n4. CHECKPOINT STRUCTURE")
    print("-" * 30)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        print(f"State dict has {len(state_dict.keys())} parameters")
        
        # Show first few keys
        print("\nFirst 5 parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {i+1}. {key}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # 5. Test compatibility with each model
    print("\n5. MODEL COMPATIBILITY TEST")
    print("-" * 30)
    
    compatible_models = []
    
    for model_type in ["vit_b", "vit_l", "vit_h"]:
        try:
            # Create model without checkpoint
            model = sam_model_registry[model_type](checkpoint=None)
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            missing = model_keys - checkpoint_keys
            unexpected = checkpoint_keys - model_keys
            
            if len(missing) == 0 and len(unexpected) == 0:
                print(f"✅ {model_type}: PERFECTLY COMPATIBLE")
                compatible_models.append(model_type)
            elif len(missing) == 0:
                print(f"⚠️  {model_type}: Compatible (has extra keys: {len(unexpected)})")
                compatible_models.append(model_type)
            elif len(unexpected) == 0:
                print(f"⚠️  {model_type}: Missing keys: {len(missing)}")
            else:
                print(f"❌ {model_type}: Missing: {len(missing)}, Extra: {len(unexpected)}")
                
        except Exception as e:
            print(f"❌ {model_type}: Error - {e}")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 30)
    
    if compatible_models:
        print(f"✅ Your checkpoint is compatible with: {compatible_models}")
        print(f"Recommended model to use: {compatible_models[0]}")
        
        # Show usage example
        print(f"\nUsage example:")
        print(f"sam = sam_model_registry['{compatible_models[0]}'](checkpoint='{checkpoint_path}')")
        
    else:
        print("❌ No compatible models found!")
        print("\nTry these solutions:")
        print("1. Download official checkpoints:")
        
        official_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        for model_type, url in official_urls.items():
            print(f"   {model_type}: {url}")
            
        print("\n2. Update segment-anything package:")
        print("   pip install --upgrade segment-anything")
        
        print("\n3. Try loading with strict=False:")
        print("   sam = sam_model_registry['vit_b'](checkpoint=None)")
        print("   sam.load_state_dict(torch.load(checkpoint_path), strict=False)")

# Usage
if __name__ == "__main__":
    # Replace with your checkpoint path
    checkpoint_path = "/Users/akashsaha/Downloads/sam_vit_h_4b8939.pth"
    check_sam_compatibility(checkpoint_path)