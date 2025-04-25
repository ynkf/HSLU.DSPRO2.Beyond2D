# Depth Estimation
// todooo

# Monocular Depth Estimation

## Classical SfM:

- Depth is computed through multi-view geometry
- The pipeline typically follows: feature matching → camera pose estimation → triangulation → bundle adjustment
- At least two views are required to **establish depth through triangulation**
- No initial depth estimates from single images are needed

Monocular depth estimation is technically redundant because:

1. SfM inherently recovers depth through triangulation from multiple views
2. Traditional SfM doesn't require any initial depth estimates
3. The scale ambiguity in monocular depth would need to be resolved anyway

## Modern SfM
Despite not being necessary, monocular depth estimation offers several advantages:

### Initialization and Convergence

- Provides good initial estimates that can help SfM converge faster
- Can break ambiguities in challenging scenes


### Sparse-to-Dense Pipeline Enhancement

- Traditional SfM produces sparse point clouds
- Monocular depth can help densify these reconstructions


### Handling Challenging Scenarios

- Helps with textureless regions where feature matching struggles
- Aids reconstruction of scenes with repetitive patterns


### Scale Information

- Can provide relative depth information when absolute scale isn't known
- Helps with consistency in scene reconstruction


### Hybrid Approaches

- COLMAP and other modern SfM systems increasingly incorporate learning-based components
- Monocular depth cues can complement geometric constraints