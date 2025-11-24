# Capstone-From-2D-Civil-Floor-Drawings-to-3D-Models
This project builds a semi-automated pipeline that converts 2D civil floor plans into interactive 3D models using computer vision, segmentation models (UNet), and 3D geometry reconstruction.

### Key Features

- Image Segmentation (UNet)
Detects walls, doors, windows, and rooms from scanned 2D drawings.

- Polygon Extraction & Cleaning
Uses geometry libraries (Shapely) to convert segmented masks into valid 2D polygons.

- Automated 3D Model Generation
Extrudes 2D polygons into 3D meshes using Trimesh.

- Realistic Coloring & Transparency
Semi-transparent walls, door openings, window frames, customizable materials.

- Export to GLB/OBJ
Outputs 3D models ready for visualization in browsers, Blender, or AR/VR viewers.

- Fast Batch-Processing Pipeline
Process thousands of plans with a single command.

### Tech Stack

Python, OpenCV, NumPy

UNet Segmentation Model

Shapely for geometric reconstruction

Trimesh / PyVista for 3D modeling & visualization

GLTF/GLB Exporter

### Project Aim

To bridge traditional 2D civil engineering drawings with modern 3D/AR visualization, enabling architects, engineers, and construction teams to understand plans faster and more accurately.

### Status

-- Working pipeline
-- Model trained
-- 3D visualization working (with coloring)
-- Improving height normalization & color customization
-- Web-based uploader → auto 3D viewer
