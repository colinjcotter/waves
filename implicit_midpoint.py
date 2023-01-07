from firedrake import *

n = 200
mesh = UnitIntervalMesh(10)
m = 200
mesh = ExtrudedMesh(mesh, m, layer_height=1/m)

V = FunctionSpace(mesh, "CG", 1)
T = FunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)

W = T*V

w0 = Function(W)
w1 = Function(W)
x, y = SpatialCoordinate(mesh)

# Using a time staggered scheme. phi is DG0 in time,
# eta is CG1 in time.

# \int_{t_{n-1}}^{t_{n+1}} <<\eta_t, \phi>> + g<< \eta, \eta >> 
# + 0.5*< phi, phi > dt

eta0, phi0 = split(w0)
eta1, phi1 = split(w1)
etah = 0.5*(eta0 + eta1)
phih = 0.5*(phi0 + phi1)

deta, dphi = TestFunctions(W)

dt = Constant(0.01)
g = Constant(1.0)


#build the system to solve
eqn = (
    deta*(-phi1-phi0 + dt*g*etah)*ds_t
    + dphi*(eta1-eta0)*ds_t
    + dt*inner(grad(phih), grad(dphi))*dx
)

# build the preconditioning operator
eta, phi = TrialFunctions(W)
eta1 = phi*2/dt/g
Jp = (
    deta*g*eta/2*ds_t
    #find this part by eliminating eta1 and ignoring all RHS bits
    + dphi*phi*2/dt/g*ds_t
    + dt*0.5*inner(grad(phi), grad(dphi))*dx
    )

prob = NonlinearVariationalProblem(eqn, w1, Jp=Jp)
params = {
    'snes_monitor': None,
    'ksp_type':'gmres',
    'ksp_monitor': None,
    'pc_type':'fieldsplit',
    'pc_fieldsplit_type': 'multiplicative',
    'fieldsplit_0_pc_type': 'lu',
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'lu',
    'fieldsplit_1_ksp_type': 'preonly',
}



solver = NonlinearVariationalSolver(prob, solver_parameters=params)

# set an initial condition
eta0, phi0 = w0.split()

eta0.interpolate(cos(x)*sin(y))

solver.solve()
