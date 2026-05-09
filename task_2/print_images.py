import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

def generate_dataset_proof_grid(
    data_dir="task_2/data/golden_dataset2/golden_dataset2", 
    samples_per_class=8, 
    save_path="appendix_golden_dataset2_proof.png"
):
    """
    Generates a compact grid of square-cropped images.
    Rows = Produce Type, Columns = Random Image Samples.
    """
    base_path = Path(data_dir)
    classes = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"No directories found in {data_dir}. Check your path!")
        return

    num_classes = len(classes)
    
    # Reduced figsize for a more compact, lower-res layout
    fig, axes = plt.subplots(
        nrows=num_classes, 
        ncols=samples_per_class, 
        figsize=(1.2 * samples_per_class, 1.2 * num_classes)
    )
    
    if num_classes == 1:
        axes = [axes]

    for i, cls_dir in enumerate(classes):
        cls_name = cls_dir.name
        
        all_images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        
        if len(all_images) >= samples_per_class:
            sampled_images = random.sample(all_images, samples_per_class)
        else:
            sampled_images = all_images
            
        for j in range(samples_per_class):
            ax = axes[i][j]
            
            # Remove axis ticks and borders
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if j < len(sampled_images):
                try:
                    img = Image.open(sampled_images[j]).convert("RGB")
                    
                    # --- CENTER CROP TO SQUARE ---
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) / 2
                    top = (height - min_dim) / 2
                    right = (width + min_dim) / 2
                    bottom = (height + min_dim) / 2
                    
                    # Crop and resize to low-res thumbnail (128x128)
                    img = img.crop((left, top, right, bottom)).resize((128, 128))
                    
                    ax.imshow(img)
                except Exception as e:
                    print(f"Error loading {sampled_images[j]}: {e}")
            
            # Label the row on the far left column
            if j == 0:
                formatted_name = cls_name.replace("_", " ").title()
                ax.set_ylabel(
                    formatted_name, 
                    rotation=0, 
                    size=12, # Slightly smaller font to match new scale
                    weight='bold', 
                    labelpad=40,
                    ha='right',
                    va='center'
                )

    plt.tight_layout()
    plt.subplots_adjust(left=0.25) # Extra room for the row labels
    
    # Save at lower DPI (150) to keep file sizes manageable
    print(f"Saving square, lower-res grid to {save_path}...")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

# Run the function
generate_dataset_proof_grid()