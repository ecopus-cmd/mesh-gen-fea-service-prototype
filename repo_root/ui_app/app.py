import os
import tempfile
import subprocess
import traceback

import numpy as np
import meshio
import gradio as gr
import plotly.graph_objects as go

import requests

# URL for the remote FEniCS microservice (can be overridden via env)
FEA_SERVICE_URL = os.environ.get(
    "FEA_SERVICE_URL", "http://fenics-fea-service:8000/solve_elasticity"
)
# For local testing without Docker compose, you can set:
#   FEA_SERVICE_URL="http://host.docker.internal:8000/solve_elasticity"
# or override via:  docker run -e FEA_SERVICE_URL=... mesh-demo


# -------------------------------
# Logger
# -------------------------------
def log(msg: str):
    print(msg, flush=True)


# -------------------------------
# Material database
# -------------------------------
MATERIALS = {
    "Steel": {"E": 210e9, "nu": 0.3},
    "Aluminum": {"E": 70e9, "nu": 0.33},
    "Titanium": {"E": 110e9, "nu": 0.34},
}


# -------------------------------
# Extract STEP path
# -------------------------------
def get_step_path(step_file):
    if hasattr(step_file, "name"):
        return step_file.name
    if isinstance(step_file, str):
        return step_file
    if isinstance(step_file, dict) and "name" in step_file:
        return step_file["name"]
    raise TypeError(f"Unsupported file type: {type(step_file)}")

# -------------------------------
# Helper: Decode Base64 Arrays
# -------------------------------
def decode_array(b64_string, dtype=np.float64):
    if not b64_string:
        return np.array([])
    data = base64.b64decode(b64_string)
    return np.frombuffer(data, dtype=dtype)

# -------------------------------
# Utility: build surface triangle list from tetrahedra
# -------------------------------
def extract_surface_triangles(tet_cells):
    """
    Given an (N,4) array of tet node indices, return (M,3) array of
    boundary triangle node indices (each face that belongs to only one tet).
    """
    face_dict = {}
    faces_local = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )

    for tet in tet_cells:
        for fl in faces_local:
            face = tuple(sorted(tet[fl].tolist()))
            face_dict[face] = face_dict.get(face, 0) + 1

    surface_faces = [f for f, c in face_dict.items() if c == 1]
    if not surface_faces:
        return np.zeros((0, 3), dtype=int)

    return np.array(surface_faces, dtype=int)


# -------------------------------
# Step 1: Mesh generation (no force)
# -------------------------------
def generate_mesh(step_file, material_name, mesh_size):
    try:
        log("=== Step 1: Generating mesh ===")

        if step_file is None:
            return "Error: No STEP file provided.", "failed", None, None, None

        # Clean STEP file path
        step_path = get_step_path(step_file)

        import re

        step_path = re.sub(
            r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e]",
            "",
            step_path,
        )

        if not os.path.isfile(step_path):
            return f"Error: STEP file not found:\n{step_path}", "failed", None, None, None

        log(f"STEP file path: {step_path}")

        # Material (stored for later FEA)
        mat = MATERIALS.get(material_name, MATERIALS["Steel"])
        E = mat["E"]
        nu = mat["nu"]

        # Workspace
        tmpdir = tempfile.mkdtemp(prefix="mesh-demo-")
        clean_step = os.path.join(tmpdir, "input.step")
        msh_path = os.path.join(tmpdir, "mesh.msh")
        geo_path = os.path.join(tmpdir, "model.geo")

        # Copy STEP into clean path
        with open(step_path, "rb") as src, open(clean_step, "wb") as dst:
            dst.write(src.read())

        clean_step_unix = clean_step.replace("\\", "/")
        msh_path_unix = msh_path.replace("\\", "/")

        # Build .geo content (this variant works with Gmsh 4.11.1)
        geo_lines = [
            'SetFactory("OpenCASCADE");',
            f'Merge "{clean_step_unix}";',

            # REQUIRED physical group definitions
            'Physical Volume("volume") = {1};',
            'Physical Surface("surface") = {1};',

            f'CharacteristicLengthMin = {mesh_size};',
            f'CharacteristicLengthMax = {mesh_size};',
            "Mesh 3;",
            f'Save "{msh_path_unix}";',
        ]

        geo_script = "\n".join(geo_lines) + "\n"

        log(f"DEBUG geo_script repr(): {repr(geo_script)}")

        with open(geo_path, "wb") as f:
            f.write(geo_script.encode("ascii"))

        log("------ BEGIN .geo ------")
        with open(geo_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                log(f"{i:02d}: {line.rstrip()}")
        log("------ END .geo ------")

        # Run Gmsh
        cmd = ["gmsh", geo_path, "-3", "-format", "msh4", "-nopopup", "-v", "2"]
        log("Running Gmsh: " + " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            log("Gmsh stderr:\n" + result.stderr)
            return "Error: Gmsh failed.\n" + result.stderr, "failed", None, None, None

        if not os.path.isfile(msh_path):
            return "Error: No mesh generated.", "failed", None, None, None

        log(f"Mesh created: {msh_path}")

        # Read mesh
        mesh = meshio.read(msh_path)
        points = np.asarray(mesh.points)
        cells_dict = {}
        for block in mesh.cells:
            cells_dict.setdefault(block.type, []).append(block.data)

        tet_cells = None
        cell_type = None
        for k in ("tetra", "tetra10"):
            if k in cells_dict:
                tet_cells = np.vstack(cells_dict[k])
                cell_type = k
                break

        if tet_cells is None:
            return (
                f"Error: No tetrahedral cells found. Found: {list(cells_dict.keys())}",
                "failed",
                None,
                None,
                None,
            )

        num_nodes = len(points)
        num_cells = len(tet_cells)

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent = maxs - mins

        # Edge statistics
        edges = []
        for tet in tet_cells[: min(200, num_cells)]:
            verts = points[tet]
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append(np.linalg.norm(verts[i] - verts[j]))
        edges = np.array(edges)
        avg_edge = float(np.mean(edges))
        min_edge = float(np.min(edges))
        max_edge = float(np.max(edges))

        char_area = avg_edge**2 if avg_edge > 0 else 1.0  # avoid divide-by-zero later

        # Prepare text report (no force yet)
        report = f"""
=== Mesh Summary ===

Mesh:
  Type           : {cell_type}
  Nodes          : {num_nodes}
  Tetra cells    : {num_cells}

Bounding box:
  Min → {mins}
  Max → {maxs}
  Size → {extent}

Edge Lengths:
  Avg            : {avg_edge:.4e}
  Min            : {min_edge:.4e}
  Max            : {max_edge:.4e}

Material (for later FEA):
  Material       : {material_name}
  E              : {E:.3e}
  nu             : {nu:.3f}

No force applied yet.
Use the "Apply Force" section below to choose a location & force vector.
"""

        # Build surface triangles for visualization
        surface_tris = extract_surface_triangles(tet_cells)

        # Build Plotly figure: nodes + surface mesh (no force yet)
        fig = go.Figure()

        # Node scatter
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name="Nodes",
            )
        )

        # Surface mesh (triangles)
        if surface_tris.shape[0] > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=surface_tris[:, 0],
                    j=surface_tris[:, 1],
                    k=surface_tris[:, 2],
                    opacity=0.4,
                    color="lightblue",
                    name="Surface",
                )
            )

        fig.update_layout(
            scene=dict(aspectmode="data"),
            title="Mesh Visualization (Nodes + Surface)",
            height=700,
        )

        # Store state for step 2 (apply_force)
        mesh_state = {
            "msh_path": msh_path,
            "material_name": material_name,
            "E": E,
            "nu": nu,
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "extent": extent.tolist(),
            "char_area": char_area,
            "avg_edge": avg_edge
        }

        status = "mesh_ok"

        # Return: report, status, figure, downloadable mesh, state
        return report, status, fig, msh_path, mesh_state

    except Exception:
        tb = traceback.format_exc()
        log(tb)
        return f"Error:\n{tb}", "failed", None, None, None


# -------------------------------
# Remote FEA microservice caller
# -------------------------------
def call_remote_fea(msh_path, E, nu, Fx, Fy, Fz):
    """
    Send mesh + material + force vector to the FEniCS microservice.
    Returns a dict (with keys like 'max_displacement') or None on error.
    """
    if not os.path.isfile(msh_path):
        return None

    try:
        with open(msh_path, "rb") as f:
            files = {"mesh_file": ("mesh.msh", f, "application/octet-stream")}
            data = {
                "E": str(E),
                "nu": str(nu),
                "Fx": str(Fx),
                "Fy": str(Fy),
                "Fz": str(Fz),
            }

            resp = requests.post(FEA_SERVICE_URL, files=files, data=data, timeout=60)
            if resp.status_code != 200:
                log(f"Remote FEA HTTP error: {resp.status_code}")
                return None
            return resp.json()
    except Exception as exc:
        log(f"Remote FEA call failed: {exc}")
        return None


# -------------------------------
# Step 2: Apply a force at user location
# -------------------------------
def apply_force(mesh_state, fx, fy, fz, loc_x, loc_y, loc_z):
    try:
        log("=== Step 2: Applying force ===")

        if mesh_state is None:
            return "Error: No mesh in memory. Generate a mesh first.", "failed", None

        msh_path = mesh_state.get("msh_path")
        if not msh_path or not os.path.isfile(msh_path):
            return "Error: Stored mesh file not found. Re-run mesh generation.", "failed", None

        # Reload mesh
        mesh = meshio.read(msh_path)
        points = np.asarray(mesh.points)
        cells_dict = {}
        for block in mesh.cells:
            cells_dict.setdefault(block.type, []).append(block.data)

        tet_cells = None
        cell_type = None
        for k in ("tetra", "tetra10"):
            if k in cells_dict:
                tet_cells = np.vstack(cells_dict[k])
                cell_type = k
                break
                
        if tet_cells is None:
            return "Error: No tetrahedral cells found when reloading mesh.", "failed", None


        num_nodes = len(points)
        num_cells = len(tet_cells)

        mins = np.array(mesh_state["mins"])
        maxs = np.array(mesh_state["maxs"])
        extent = np.array(mesh_state["extent"])
        char_area = float(mesh_state["char_area"])
        material_name = mesh_state["material_name"]
        E = float(mesh_state["E"])
        nu = float(mesh_state["nu"])

        # Force vector
        fx = float(fx or 0.0)
        fy = float(fy or 0.0)
        fz = float(fz or 0.0)
        F_vec = np.array([fx, fy, fz], dtype=float)
        F_mag = float(np.linalg.norm(F_vec))

        # Location: if any of x,y,z is None, fall back to bounding-box max corner
        if loc_x is None or loc_y is None or loc_z is None:
            p0 = maxs
            location_note = "Location: bounding-box max corner (default)"
        else:
            p0 = np.array([float(loc_x), float(loc_y), float(loc_z)], dtype=float)
            location_note = (
                "Location: user-specified "
                f"[{p0[0]:.3e}, {p0[1]:.3e}, {p0[2]:.3e}]"
            )

        # Arrow direction
        if F_mag > 0:
            direction = F_vec / F_mag
            base_length = np.max(extent) * 0.3 if np.max(extent) > 0 else 1.0
            p1 = p0 + direction * base_length
        else:
            # Zero-force → tiny arrow (no direction)
            p1 = p0.copy()

        # Toy displacement estimate using |F|
        if char_area <= 0:
            char_area = 1.0
        u_est = F_mag / (E * char_area)

        # Call remote FEA microservice (optional)
        fea_result = call_remote_fea(msh_path, E, nu, fx, fy, fz)

        if fea_result and fea_result.get("status") == "ok":
            max_u = float(fea_result.get("max_displacement", float("nan")))
            avg_u = float(fea_result.get("avg_displacement", float("nan")))
            vol = float(fea_result.get("volume", float("nan")))
            fea_summary = (
                "\nRemote FEniCS FEA (microservice):\n"
                f"  max |u|        : {max_u:.4e}\n"
                f"  avg |u|        : {avg_u:.4e}\n"
                f"  domain volume  : {vol:.4e}\n"
            )
        else:
            fea_summary = (
                "\nRemote FEniCS FEA: unavailable or failed.\n"
                "Once the microservice is running, this section will show\n"
                "true FEM displacement metrics.\n"
            )

        # Build text report
        report = f"""
=== Mesh + Force Summary ===

Mesh:
  Type           : {cell_type}
  Nodes          : {num_nodes}
  Tetra cells    : {num_cells}

Bounding box:
  Min → {mins}
  Max → {maxs}
  Size → {extent}

Force:
  Vector [Fx, Fy, Fz] : [{fx:.3e}, {fy:.3e}, {fz:.3e}]
  Magnitude |F|       : {F_mag:.3e}
  {location_note}

Toy FEA Estimate (using |F| and characteristic area from mesh):
  Material       : {material_name}
  E              : {E:.3e}
  nu             : {nu:.3f}
  Displacement   : {u_est:.3e}

{fea_summary}
NOTE: The 'Toy FEA Estimate' is a very crude scalar approximation.
A proper stress/displacement field requires the full FEniCS solve.
"""

        # Surface triangles
        surface_tris = extract_surface_triangles(tet_cells)

        # Build Plotly figure: nodes + surface + force arrow
        fig = go.Figure()

        # Nodes
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name="Nodes",
            )
        )

        # Surface mesh
        if surface_tris.shape[0] > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=surface_tris[:, 0],
                    j=surface_tris[:, 1],
                    k=surface_tris[:, 2],
                    opacity=0.4,
                    color="lightblue",
                    name="Surface",
                )
            )

        # Force arrow (tail at p0, head at p1)
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines+markers",
                line=dict(width=6, color="red"),
                marker=dict(size=4, color="red"),
                name="Applied Force",
            )
        )

        # Mark the application point more clearly
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0]],
                y=[p0[1]],
                z=[p0[2]],
                mode="markers",
                marker=dict(size=6, color="orange", symbol="diamond"),
                name="Force Location",
            )
        )

        fig.update_layout(
            scene=dict(aspectmode="data"),
            title="Mesh + Applied Force",
            height=700,
        )
        print("FEA_SERVICE_URL =", FEA_SERVICE_URL)
        print("Microservice response =", fea_result)
        return report, "force_ok", fig

    except Exception:
        tb = traceback.format_exc()
        log(tb)
        return f"Error:\n{tb}", "failed", None


# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks(title="Meshing + Visualization + Force Placement") as demo:
    gr.Markdown(
        "## Meshing Prototype (Gmsh → MeshIO → Plotly)\n"
        "1. Generate a mesh from a STEP file.\n"
        "2. Apply a force at a chosen location on the part.\n"
        "3. Call a remote FEniCS microservice for a true 3D elasticity solve."
    )

    mesh_state = gr.State()

    # --- Step 1: Mesh generation ---
    gr.Markdown("### Step 1 — Generate Mesh")

    with gr.Row():
        step_input = gr.File(
            label="STEP File",
            type="filepath",
            file_types=[".step", ".stp"],
        )
        material_input = gr.Dropdown(
            list(MATERIALS.keys()),
            value="Steel",
            label="Material",
        )
        mesh_input = gr.Slider(
            0.001,
            0.1,
            value=0.01,
            step=0.001,
            label="Element Size (approx.)",
        )

    generate_btn = gr.Button("Generate Mesh", variant="primary")

    result_box = gr.Textbox(lines=20, label="Summary")
    status_box = gr.Textbox(lines=1, label="Status")

    mesh_viewer = gr.Plot(label="Mesh Viewer")
    mesh_download = gr.File(label="Download Mesh (.msh)")

    generate_btn.click(
        fn=generate_mesh,
        inputs=[step_input, material_input, mesh_input],
        outputs=[result_box, status_box, mesh_viewer, mesh_download, mesh_state],
    )

    # --- FEA explanation box ---
    gr.Markdown(
        "### What does the FEA microservice do?\n"
        "- Solves **3D linear elasticity** on the generated tetrahedral mesh using FEniCSx.\n"
        "- Applies a **clamped boundary** on the lowest Z surface (z = z_min).\n"
        "- Converts your net force vector (Fx, Fy, Fz) into an equivalent **body force**.\n"
        "- Returns global quantities:\n"
        "  - Max displacement magnitude over all DOFs\n"
        "  - Average displacement magnitude\n"
        "  - Domain volume (from the finite-element mesh)\n\n"
        "These results are summarized numerically in the text box and visually in the\n"
        "\"Displacement comparison\" bar chart below."
    )

    # --- Step 2: Apply force ---
    gr.Markdown("### Step 2 — Apply Force to Mesh")

    gr.Markdown(
        "Specify a **force vector** and (optionally) a **location** in 3D.\n\n"
        "- If you leave any of X, Y, or Z blank, the code will apply the force\n"
        "  at the *bounding-box max* corner.\n"
        "- You can read approximate coordinates off the mesh viewer.\n"
        "- The FEniCS microservice is called automatically to compute a full\n"
        "  elasticity solve whenever you apply a nonzero force."
    )

    with gr.Row():
        fx_input = gr.Number(value=0.0, label="Fx [N]")
        fy_input = gr.Number(value=0.0, label="Fy [N]")
        fz_input = gr.Number(value=-1000.0, label="Fz [N]")

    with gr.Row():
        loc_x_input = gr.Number(value=None, label="Location X (optional)")
        loc_y_input = gr.Number(value=None, label="Location Y (optional)")
        loc_z_input = gr.Number(value=None, label="Location Z (optional)")

    apply_btn = gr.Button("Apply Force", variant="secondary")

    apply_btn.click(
        fn=apply_force,
        inputs=[mesh_state, fx_input, fy_input, fz_input, loc_x_input, loc_y_input, loc_z_input],
        outputs=[result_box, status_box, mesh_viewer],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
