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
```

## NOTES
--> you must utilize a docker engine to this repo as these Python packages require a Windows Linux subsystem. 
--> docker.desktop is free and works great for this repo: https://www.docker.com/products/docker-desktop/ 
--> follow this link for a demo: https://drive.google.com/file/d/1Ki6btzYgXYet3SrkbYCvF6xwqTX7tGTx/view?usp=sharing 
