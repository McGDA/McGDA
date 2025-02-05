# umap_utils.py
import umap.umap_ as umap
from rotation_utils import rotation_matrix_nd

def learn_umap_and_rotate(X_source, y_source, n_components=32, angle_deg=45):
    """
    Learns UMAP on X_source in n_components dimensions and returns:
      - X_source_umap: UMAP features of the source domain,
      - X_target_umap: features obtained by applying a rotation in R^n_components.
    """
    umap_extractor = umap.UMAP(n_components=n_components, random_state=42)
    X_source_umap = umap_extractor.fit_transform(X_source, y_source)
    
    # Create the rotation matrix in R^n_components
    R = rotation_matrix_nd(n_components, angle_deg)
    # Apply the rotation: multiply each vector by R^T
    X_target_umap = X_source_umap.dot(R.T)
    return X_source_umap, X_target_umap