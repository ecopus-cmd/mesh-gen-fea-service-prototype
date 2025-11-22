# fea_service.py
import os
import tempfile
from typing import Optional
import traceback

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc as PETSc_mod
from dolfinx import fem
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_geometrical, assemble_scalar, form, Function
from dolfinx.fem import petsc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh   #not working, gmshio does not exist
import ufl
from ufl import TrialFunction, TestFunction, sym, grad, inner, dx, Identity, tr
import basix.ufl



app = FastAPI(title="FEniCSx FEA Microservice")


def solve_linear_elasticity(
    msh_path: str,
    E: float,
    nu: float,
    fx: float,
    fy: float,
    fz: float
):
    comm = MPI.COMM_WORLD

    # Load mesh using correct gmsh API
    #domain, cell_tags, facet_tags = gmsh.read_from_msh(msh_path, comm, 0)
    mesh_data = gmsh.read_from_msh(msh_path, comm)    # returns (mesh, …)
    domain = mesh_data.mesh  

    # ------------------------------------------------------
    # BUILD VECTOR FINITE ELEMENT SPACE
    # (UFL 2025 – use ufl.FiniteElement with shape=)
    # ------------------------------------------------------
    element = basix.ufl.element(
        "Lagrange",
        domain.ufl_cell().cellname(),
        degree=1,
        shape=(domain.geometry.dim,)      # vector field element
    )

    V = functionspace(domain, element)

    # Trial and test functions
    u = Function(V)
    v = TestFunction(V)
    du = TrialFunction(V)

    # ------------------------------------------------------
    # MATERIAL PARAMETERS
    # ------------------------------------------------------
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return 2.0 * mu * epsilon(w) + lmbda * tr(epsilon(w)) * Identity(len(w))


    x = domain.geometry.x
    z_min = domain.comm.allreduce(np.min(x[:,2]), op=MPI.MIN)

    def clamped_boundary(x):
        return np.isclose(x[2], z_min, atol=1e-8)

    boundary_dofs = locate_dofs_geometrical(V, clamped_boundary)

    u_D = Function(V)
    u_D.x.array[:] = 0.0
    bc = dirichletbc(u_D, boundary_dofs)

    # Compute volume
    from dolfinx.fem import assemble_scalar
    volume_form = form(Constant(domain,1.0) *dx)
    volume_local = assemble_scalar(volume_form)
    volume = domain.comm.allreduce(volume_local, op=MPI.SUM)

    if volume <= 0:
        raise RuntimeError("Computed zero or negative volume from mesh.")

    f_vec = np.array([fx, fy, fz], dtype=np.float64) / volume
    f = Constant(domain, f_vec)

    a = form(inner(sigma(du), epsilon(v)) * dx)
    L = form(inner(f, v) * dx)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="elasticity", u=u)
    u = problem.solve()

    u_array = u.x.array.reshape((-1, domain.geometry.dim))
    disp_mag = np.linalg.norm(u_array, axis=1)

    max_disp = float(np.max(disp_mag))
    avg_disp = float(np.mean(disp_mag))

    return {
        "max_displacement": max_disp,
        "avg_displacement": avg_disp,
        "volume": volume,
    }


@app.post("/solve_elasticity")
async def elasticity_endpoint(
    mesh_file: UploadFile = File(...),
    E: float = Form(...),
    nu: float = Form(...),
    Fx: float = Form(...),
    Fy: float = Form(...),
    Fz: float = Form(...),
):
    try:
        with tempfile.TemporaryDirectory(prefix="fenics-mesh-") as tmpdir:
            msh_path = os.path.join(tmpdir, "input.msh")
            with open(msh_path, "wb") as f:
                f.write(await mesh_file.read())

            results = solve_linear_elasticity(
                msh_path=msh_path,
                E=E,
                nu=nu,
                fx=Fx,
                fy=Fy,
                fz=Fz,
            )

        return JSONResponse({"status": "ok", **results})

    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500,
        )
