# CIVIL-3D-PIPELINE: From 2D Civil Floor Drawings to 3D Models
A Computer Vision + Geometry Reconstruction Pipeline for Automated Building Model Generation

Disclaimer :THE SAMPLE CODE, MODELS, AND PIPELINE PROVIDED IN THIS PROJECT ARE OFFERED “AS IS”, WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL SENAKSHI KRISHNA MURTHY OR ANY OTHER CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING BUT NOT LIMITED TO LOSS OF DATA, LOSS OF PROFITS, OR INTERRUPTED WORKFLOW) ARISING IN ANY WAY OUT OF THE USE OF THIS PIPELINE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. USE THIS CODE AT YOUR OWN RISK.

# SUMMARY
Traditional civil engineering drawings are predominantly represented in 2D formats (PDF/PNG/JPG). These drawings, while effective for experts, are difficult for non-technical stakeholders — project managers, clients, community boards — to visualize.

This project introduces CIVIL-3D-PIPELINE, a semi-automated system that converts 2D civil floor plans into interactive 3D models using:

- UNet neural networks for segmentation
- Geometric polygon reconstruction
- Automated 3D mesh extrusion
- GLB/OBJ export for AR/VR or web visualization

Unlike manual CAD modeling (slow, expensive, error-prone), this pipeline accelerates floor-plan interpretation, reduces modeling time, and enhances stakeholder communication.

- The pipeline detects and reconstructs:
- Walls (extruded 3D geometry)
- Doors (cut-out openings)
- Windows (sill & head-level geometry)
- Rooms (floor slabs)

The final model is stylized with semi-transparent walls, adjustable color themes, and export-ready 3D meshes.

# Datasets Used
(User collected a mixed dataset; list can be customized)
- Hand-collected CAD-to-PNG civil plan dataset
- Public floor-plan datasets (R-FP, CubiCasa5k segments)
- Weak-labeled pseudo masks generated from contour-based algorithms

# Key Contributions
1. Fully Automated Computer Vision Pipeline
From raw floor plan → segmentation → polygon extraction → 3D mesh.
2. UNet-based Floor Plan Segmentation
Trained on thousands of annotated/mechanically labeled plans.
3. Geometry Reconstruction Engine
Shapely-powered polygon extraction and cleaning.
4. 3D Extrusion System
Wall height, window sill/head height, door opening logic.
5. Web UI (Flask) Integration
Upload → Process → Preview 3D Model in Browser.
6. AR/VR Ready GLB Output
Models are directly compatible with Three.js, Babylon.js, and Blender.

# Pipeline Architecture
1. Input Processing
- Accepts .png, .jpg, .pdf (converted to image)
- Preprocessing includes resizing, denoising, inversion, contour smoothing

2. UNet Segmentation

- Classes:
0 → Background
1 → Walls
2 → Doors
3 → Windows
4 → Rooms

Output: a segmentation mask with class-separated channels.

3. Polygon Extraction

- Using Shapely, the system:
- Converts mask → contours → polygons
- Removes noise / small polygons
- Repairs invalid geometry using buffer(0)
- Merges polygons when appropriate (unary_union)

4. 3D Model Generation

- Using Trimesh:
- Walls = extruded polygons
- Doors = boolean subtract operations
- Windows = raised slabs between sill and head height
- Rooms = flat floor slabs

Optional features:

- Wall color presets (teal / brown / grey)
- Adjustable transparency
- Background color switching

5. Export

Outputs:

.glb
.obj
.ply
.stl

These can be loaded into:

Blender
WebGL/Three.js viewer
Unity/Unreal
Mobile AR frameworks

# Models Used
1. Segmentation Models

- UNet (Baseline)
- UNet (Improved with Encoder: ResNet34)
- UNet++ (Experimental)

2. Geometry / Reconstruction

- Shapely
- OpenCV
- Trimesh

3. Visualization
- PyVista
- pygltflib (material editing)
- Three.js (web viewer)

# Installing & Running the Pipeline
1. Clone the Repo git clone - https://github.com/Senaakshi/Capstone-From-2D-Civil-Floor-Drawings-to-3D-Models.git
2. Install Requirements - pip install -r requirements.txt
3. Running the UNet  - python train_unet_improved.py
4. Running Infer - python infer_unet_improved.py
5. Converting to 3D - python to3d_improved_colour.py
6. Running the web app - python app_2.py
7. Directory Structure - /web_uploads/          → user uploads.   
                         /web_pred/             → segmentation masks. 
                         /out_3d/               → generated 3D GLB models. 
                         /unet_model/           → trained .pth UNet file. 
                         /to3d_improved_colour/ → 3D conversion scripts
8. Height & Color Customization - Inside to3d_improved_colour.py:
                                    WALL_H = 3.0          # meters. 
                                    DOOR_H = 2.1. 
                                    WIN_SILL = 0.9. 
                                    WIN_HEAD = 2.1. 
                                    COLOR_WALL = [210, 140, 80, 160]  # light brown, semi-transparent

# Future Work

- Height normalization dataset
- Multi-floor support
- Material realism (PBR textures)
- True CAD line-thickness interpretation
- Point-cloud export
- Integration with HoloLens / Vision Pro                                    


