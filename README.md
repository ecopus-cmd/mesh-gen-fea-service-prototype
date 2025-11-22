# mesh-gen-fea-service-prototype
This is a repo which deploys mesh generation (gmsh) and a FEA service (FEniCS) via a Docker engine to a Gradio GUI

# AI Mesh Optimizer Demo (FEniCSx + Gmsh App)

This repository is a deployable Docker Space that:
- Loads a STEP CAD file
- Generates a tetrahedral mesh via Gmsh
- Converts it into a FEniCSx mesh
- Runs a very simple linear elasticity solve
- Returns results interactively via Gradio

## Local Testing (Docker)
--> run at repo_root> level in terminal
```bash
docker compose build 
docker compose up

