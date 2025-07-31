import os
import nibabel as nib
import matplotlib.pyplot as plt

def save_segmentation_visualization(t2_path, organ_path, lesion_path, output_dir, slice_index=None):
    """
    Save T2 image, organ segmentation, and lesion segmentation visualizations.

    Args:
        t2_path (str): Path to T2-weighted NIfTI image.
        organ_path (str): Path to organ segmentation NIfTI image.
        lesion_path (str): Path to lesion segmentation NIfTI image.
        output_dir (str): Directory where output images will be saved.
        slice_index (int, optional): Axial slice index to visualize. Defaults to center slice.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load NIfTI images
    t2_img = nib.load(t2_path)
    organ_seg = nib.load(organ_path)
    lesion_seg = nib.load(lesion_path)

    # Get data arrays
    t2_data = t2_img.get_fdata()
    organ_data = organ_seg.get_fdata()
    lesion_data = lesion_seg.get_fdata()

    # Determine slice to visualize
    if slice_index is None:
        slice_index = t2_data.shape[2] // 2

    # Save individual slices
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(t2_data[:, :, slice_index].T, cmap="gray")
    plt.title("T2 Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(organ_data[:, :, slice_index].T, cmap="gray")
    plt.title("Organ Segmentation")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(lesion_data[:, :, slice_index].T, cmap="gray")
    plt.title("Lesion Segmentation")
    plt.axis("off")

    individual_path = os.path.join(output_dir, "individual_views.png")
    plt.tight_layout()
    plt.savefig(individual_path, bbox_inches="tight")
    plt.close()

    # Save overlay image
    plt.figure(figsize=(5, 5))
    plt.imshow(t2_data[:, :, slice_index].T, cmap="gray")
    plt.imshow(organ_data[:, :, slice_index].T, cmap="jet", alpha=0.5)
    plt.imshow(lesion_data[:, :, slice_index].T, cmap="hot", alpha=0.5)
    plt.title("T2 with Organ and Lesion")
    plt.axis("off")

    overlay_path = os.path.join(output_dir, "overlay_segmentation.png")
    plt.tight_layout()
    plt.savefig(overlay_path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualizations to:\n- {individual_path}\n- {overlay_path}")


save_segmentation_visualization(
    t2_path="output/t2/t2.nii.gz",
    organ_path="output/organ/organ.nii.gz",
    lesion_path="output/lesion/lesion_mask.nii.gz",
    output_dir="visualizations"
)