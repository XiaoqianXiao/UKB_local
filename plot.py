#%%
import nibabel as nib

# Path to the CIFTI file
file_path = "/Users/xiaoqianxiao/tool/parcellation/Tian2020MSA/3T/Cortex-Subcortex/Gordon333.32k_fs_LR_Tian_Subcortex_S2.dlabel.nii"

# Load the atlas image
atlas_img = load_img(atlas_path)

# Plot the atlas using nilearn
plotting.plot_roi(
    atlas_img,
    title="Brain Parcellation - Tian's Atlas",
    display_mode="ortho",  # Orthogonal slices ('ortho', 'x', 'y', 'z')
    cmap="tab20",          # Use a colormap for distinct regions
    colorbar=True
)

# Show the plot
plotting.show()

