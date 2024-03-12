import meshio
mesh_io = meshio.read("globe2.xdmf")
data = mesh_io.point_data["elevation"]

import fenics as fn
import mshr
from ufl import grad, dot, dx, ds, Max, Min
import numpy as np
from math import pi, cos, sin, sqrt

def to_rad(deg):
    return deg * pi / 180

mesh_path = "globe2.xdmf"

mesh_file = fn.XDMFFile(mesh_path)
mesh = fn.Mesh()
mesh_file.read(mesh)


global_normal = fn.Expression(("x[0]", "x[1]", "x[2]"), degree=1)
mesh.init_cell_orientations(global_normal)

# Time stepping parameters
T = 1.0e6
num_steps = 400
dt = T / num_steps
k = fn.Constant(dt)

# Create finite element space
V = fn.FunctionSpace(mesh, 'Lagrange', 1)
u_h = fn.Function(V, name='level')
v_h = fn.Function(V)
a_h = fn.Function(V)
u_prev = fn.Function(V)
u_pprev = fn.Function(V)
z = fn.Function(V, name="elevation")

depth = fn.Function(V, name='depth')
coeff = fn.Function(V, name='coeff')

# Adjust shore
for vs in mesh.cells():
    zs = np.array([data[v] for v in vs])
    if any(zs > 0) and any(zs < 0):
        for v in vs:
            if data[v] < 0:
                data[v] = 0

# set terrain data
dof_to_vertex = fn.dof_to_vertex_map(V)
z.vector()[:] = data[dof_to_vertex]

base_level = fn.Function(V)
base_level.interpolate(fn.Expression('max(0.0,z)', z=z, degree=1))

# Initial state
deg = 4

# Impact data
lattitude = 28
longitude = -65
max_h = 20
radius = 0.7e6

theta = to_rad(lattitude)
phi = to_rad(longitude)
R = 6.371e6   # Earth radius
c = fn.Constant((R * cos(phi) * cos(theta), R * sin(phi) * cos(theta), R * sin(theta)))

r = fn.Expression('sqrt(pow(x[0] - c[0], 2) + pow(x[1] - c[1], 2) + pow(x[2] - c[2], 2))', degree=deg, c=c)
lattitude_ = fn.Expression('180.0 / M_PI * asin(x[2] / R)', degree=deg, R=R)
longitude_ = fn.Expression('180.0 / M_PI * atan2(x[1], x[0])', degree=deg)
impact = fn.Expression('s * exp(-pow(r/r0, 2))', r=r, r0=radius, s=max_h, degree=deg)
blob1 = fn.Expression('(51 < lat && lat < 59.5 && -2.5 < lon && lon < 33) ? 0.0 : 6.0',
        degree=deg, lat=lattitude_, lon=longitude_)
blob2 = fn.Expression('(59 < lat && 15 < lon && lon < 33) ? 0.0 : 6.0',
        degree=deg, lat=lattitude_, lon=longitude_)
u0 = fn.Expression('min(blob1, blob2)', degree=deg, blob1=blob1, blob2=blob2)
u_h.interpolate(u0)
depth.vector()[:] = u_h.vector() - base_level.vector()
u_prev.vector()[:] = u_h.vector()

# Forcing
f = fn.Constant(0)

g = fn.Constant(9.81)

# Define forms
a = fn.TrialFunction(V)
v = fn.TestFunction(V)

# Parameters
rho = 0.2 # 0.5
alpha_f = rho / (rho + 1)
alpha_m = (2 * rho - 1) / (rho + 1)
gamma = 0.5 - alpha_m + alpha_f
beta = 0.25 * (1 - alpha_m + alpha_f) ** 2

_dt = fn.Constant(dt)
_alpha_f = fn.Constant(alpha_f)
_alpha_m = fn.Constant(alpha_m)
_gamma = fn.Constant(gamma)
_beta = fn.Constant(beta)

target_depth = 100.0
L0 = 2*3
zeta = 0.5

C0 = fn.Constant(0.0005)
C = - C0 * 5.0 / Min(-5.0, z)

# # Step form
u_new = u_h + k * v_h + k ** 2 * ((0.5 - _beta) * a_h + _beta * a)
v_new = v_h + k * ((1 - _gamma) * a_h + _gamma * a)
a_mid = (1 - _alpha_m) * a + _alpha_m * a_h
v_mid = (1 - _alpha_f) * v_new + _alpha_f * v_h
u_mid = (1 - _alpha_f) * u_new + _alpha_f * u_h
F = a_mid * v + C * v_mid * v + g * Max(10.0, u_h - z) * dot(grad(u_mid), grad(v))
b, L = fn.system(F * dx)  # FEniCS can separate F into left- and right-hand side

out = fn.XDMFFile('tsunami.xdmf')
out.parameters['flush_output'] = True              # to see partial results during simulation
out.parameters['rewrite_function_mesh'] = False    # mesh does not change between time steps
out.parameters['functions_share_mesh'] = True

W = fn.VectorFunctionSpace(mesh, 'Lagrange', 1)
E = fn.Function(W, name='terrain-grad')
fn.project(grad(z), W, function=E)
out.write(E, t=0)

grad_u = fn.Function(W, name='level-grad')
fn.project(grad(u_h), W, function=grad_u)
out.write(grad_u, t=0)

out.write(z, t=0)
out.write(u_h, t=0)
out.write(depth, t=0)
print("Starting")

# Time stepping loop
for n in range(1, num_steps+1):
    t = n * dt
    print('Step {}, t = {:.2f}'.format(n, t))

    a_new = fn.Function(V)
    fn.solve(b == L, a_new)

    _u_h = u_h.vector()
    _v_h = v_h.vector()
    _a_h = a_h.vector()

    _a_new = a_new.vector()
    _u_new = _u_h + dt * _v_h + dt ** 2 * ((0.5 - beta) * _a_h + beta * _a_new)
    _v_new = _v_h + dt * ((1 - gamma) * _a_h + gamma * _a_new)

    _a_h[:] = _a_new
    _v_h[:] = _v_new
    _u_h[:] = _u_new

    depth.vector()[:] = u_h.vector() - base_level.vector()
    out.write(u_h, t)
    out.write(depth, t)
    out.write(z, t)

    fn.project(grad(u_h), W, function=grad_u)
    out.write(grad_u, t)

    total_u = fn.assemble(u_h * dx) / (4 * pi * R**2)
    print(f'total u = {total_u:.5g}')
