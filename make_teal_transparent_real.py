from pygltflib import GLTF2, PbrMetallicRoughness

# Paths
in_path  = "/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Sena_PROJECT/FINAL/out_3d/house.glb"
out_path = "/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Sena_PROJECT/FINAL/out_3d/house_teal_transparent.glb"

gltf = GLTF2().load(in_path)

for mat in gltf.materials:
    # Ensure PBR exists
    if mat.pbrMetallicRoughness is None:
        mat.pbrMetallicRoughness = PbrMetallicRoughness()

    # Set TEAL color with 80% transparency
    # baseColorFactor = [R, G, B, A]
    # A = 0.2 → 80% see-through
    mat.pbrMetallicRoughness.baseColorFactor = [0.0, 0.8, 0.8, 0.2]

    # Enable transparent rendering
    mat.alphaMode = "BLEND"
    mat.doubleSided = True  # helps display from inside/outside walls

gltf.save(out_path)
print("✅ Saved:", out_path)

