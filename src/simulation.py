# fenics_absorbing_n_equal_circles_rect_volume_loss.py
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import math
import csv  # <--- NEW

# -----------------------------
# Physical parameters
# -----------------------------
D   = 1.0
Q   = 0.076
rho = 0.899

# NEW: area-proportional loss coefficient (see volume balance below)
q   = 0.042  # NB q is meant to be 4*q because of surface area

T_final   = 10000.0
dt        = 0.2
num_steps = int(T_final/dt)
viz_every = 5000

# -----------------------------
# Kelvin / Gibbs–Thomson boundary value: C_eq(R) = C_sat*(1 + Gamma/R)
# -----------------------------
C_sat = 0.1
Gamma = 0.10
def C_eq_of_R(R):
    R_eff = max(float(R), 1e-10)
    return float(C_sat * (1.0 + Gamma / R_eff))

# -----------------------------
# Rectangle domain
# -----------------------------
l = 10.0
r = 2.0
xL, xR = -l, l
yB, yT = -math.pi*r, math.pi*r

# -----------------------------
# Equal circles, random placement (C=0 on each circle)
# -----------------------------
rng_seed = None
rng = np.random.default_rng(rng_seed)

n_circles    = 20
a0           = 0.50
sep_margin   = 1.0
batch_factor = 60

def place_equal_circles(n, a, margin, batch_k):
    centers = []
    min_center_dist = 2.0*a + margin

    def enough_room(cx, cy):
        for (px, py) in centers:
            if math.hypot(cx - px, cy - py) < min_center_dist:
                return False
        return True

    tries = 0
    while len(centers) < n:
        tries += 1
        Xcand = rng.uniform(xL + a, xR - a, size=batch_k*n)
        Ycand = rng.uniform(yB + a, yT - a, size=batch_k*n)
        for cx, cy in zip(Xcand, Ycand):
            if enough_room(float(cx), float(cy)):
                centers.append((float(cx), float(cy)))
                if len(centers) == n:
                    break
        if tries > 2000:
            raise RuntimeError("Failed to place all circles; try fewer circles "
                               "or reduce a0/sep_margin, or enlarge batch_factor.")
    return centers

centers = place_equal_circles(n_circles, a0, sep_margin, batch_factor)
radii   = [a0]*n_circles

print("\nPlaced equal-radius circles (C=0):")
for i, (c) in enumerate(centers, start=1):
    print(f"  #{i}: center=({c[0]:.3f},{c[1]:.3f}), a={a0:.3f}")

# -----------------------------
# Build domain: rectangle minus circles
# -----------------------------
rect   = Rectangle(Point(xL, yB), Point(xR, yT))
domain = rect
for (cx, cy) in centers:
    domain -= Circle(Point(cx, cy), a0)

mesh_resolution = 120
mesh = generate_mesh(domain, mesh_resolution)

# -----------------------------
# Function space
# -----------------------------
V = FunctionSpace(mesh, "P", 1)

# -----------------------------
# Boundary tagging
# -----------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundaries.set_all(0)
n = FacetNormal(mesh)

hmin = mesh.hmin()
tol  = 2.0*hmin
outer_id = n_circles + 1

def dist_to_circle_boundary(xm, c, a):
    return abs(math.hypot(xm.x()-c[0], xm.y()-c[1]) - a)

for f in facets(mesh):
    if f.exterior():
        xm = f.midpoint()
        dists = [dist_to_circle_boundary(xm, c, a0) for c in centers]
        dmin  = min(dists) if dists else 1e9
        if dmin <= tol:
            which = int(np.argmin(dists))  # 0..n-1
            boundaries[f] = which + 1      # circle ids: 1..n
        else:
            boundaries[f] = outer_id

ds_ = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Diagnostics
for i in range(1, n_circles+1):
    L_meas = assemble(Constant(1.0)*ds_(i))
    print(f"[check] circle {i} length: {L_meas:.6f} (expected ~ {2*math.pi*a0:.6f})")

# -----------------------------
# Variational forms (implicit Euler)
# -----------------------------
u   = TrialFunction(V)
v   = TestFunction(V)
u_n = Function(V); u_n.assign(Constant(1.0))

D_c, Q_c, dt_c = Constant(D), Constant(Q), Constant(dt)
a_form = u*v*dx + dt_c*D_c*dot(grad(u), grad(v))*dx
L_form = (u_n + dt_c*Q_c)*v*dx

A = assemble(a_form)

# -----------------------------
# Boundary conditions: C = C_eq(R_j) on every circle
# -----------------------------
R0_init      = 1.5
C_inner_list = [Constant(C_eq_of_R(R0_init)) for _ in range(n_circles)]
bcs          = [DirichletBC(V, C_inner_list[i], boundaries, i+1) for i in range(n_circles)]
for bc in bcs:
    bc.apply(A)

u_sol = Function(V)

# -----------------------------
# 3D droplet bookkeeping
# -----------------------------
R0   = 1.5
R    = [R0 for _ in range(n_circles)]
V3D0 = (4.0/3.0)*math.pi*(R0**3)
V    = [V3D0 for _ in range(n_circles)]

# -----------------------------
# History in memory
# -----------------------------
time_points   = []
R_hist_list   = [[] for _ in range(n_circles)]
Phi_hist_list = [[] for _ in range(n_circles)]

# -----------------------------
# LIVE CSV writer for radii vs time
# -----------------------------
csv_filename = "fenics_multi_equal_Rt_area_loss_live.csv"
with open(csv_filename, "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    header = ["t"] + [f"R{j+1}" for j in range(n_circles)]
    writer.writerow(header)

    # -----------------------------
    # Time loop
    # -----------------------------
    for k in range(num_steps+1):
        t = k*dt

        b = assemble(L_form)
        for bc in bcs:
            bc.apply(b)
        solve(A, u_sol.vector(), b, "lu")

        # Per-circle flux and volume update
        for i in range(n_circles):
            Phi_i = assemble(-D_c*dot(grad(u_sol), n)*ds_(i+1))
            Phi_i = max(0.0, float(Phi_i))
            influx_vol_rate = Phi_i / rho

            loss_rate = math.pi * (R[i] ** 2) * q
            dVdt      = influx_vol_rate - loss_rate

            V[i] = max(0.0, V[i] + dVdt*dt)
            R[i] = (3.0*V[i]/(4.0*math.pi))**(1.0/3.0)

            R_hist_list[i].append(R[i])
            Phi_hist_list[i].append(Phi_i)

        # update BCs for next step
        for i in range(n_circles):
            C_inner_list[i].assign(C_eq_of_R(R[i]))

        time_points.append(t)

        # ---- write this time step to CSV (live) ----
        writer.writerow([t] + R[:])

        # ---- optional visualization ----
        if k % viz_every == 0:
            coords = mesh.coordinates(); cells = mesh.cells()
            fig, (axC, axR) = plt.subplots(
                1, 2, figsize=(16, 4),
                gridspec_kw={"width_ratios": [2.6, 2.6]},
                constrained_layout=True
            )

            # LEFT PANEL: concentration over full domain
            tcf = axC.tricontourf(coords[:,0], coords[:,1], cells,
                                  u_sol.compute_vertex_values(mesh),
                                  levels=60, cmap='viridis')
            axC.add_patch(plt.Rectangle((xL, yB), xR-xL, yT-yB, fill=False, lw=1))
            for j, (cx, cy) in enumerate(centers, start=1):
                axC.add_patch(plt.Circle((cx, cy), a0, color='r', fill=False, lw=1))
                axC.text(cx, cy, str(j), color='w', ha='center', va='center', fontsize=8,
                         bbox=dict(facecolor='black', alpha=0.35, boxstyle='circle,pad=0.2'))

            # 🔹 Force full spatial domain to be visible
            axC.set_xlim(xL, xR)
            axC.set_ylim(yB, yT)
            axC.set_aspect('equal', adjustable='box')

            axC.set_xlabel("x"); axC.set_ylabel("y")
            axC.set_title(f"C(x,y), t={t:.2f}")
            fig.colorbar(tcf, ax=axC, label="C")

            # RIGHT PANEL: radii vs time over full time range
            for j in range(n_circles):
                axR.plot(time_points, R_hist_list[j], lw=2, label=f"#{j+1}")

            # 🔹 Force full time range and radius range
            axR.set_xlim(0.0, T_final)
            max_R_seen = max(max(R_hist_list[j]) for j in range(n_circles) if R_hist_list[j])
            axR.set_ylim(0.0, 1.1*max_R_seen if max_R_seen > 0 else 1.0)

            axR.set_xlabel("t"); axR.set_ylabel("R(t)")
            axR.set_title("3D droplet radii with Lipolysis")
            axR.legend(loc='best', fontsize=9, frameon=True)

            plt.show()

        u_n.assign(u_sol)

print(f"[live-saved] {csv_filename}")

# -----------------------------
# Final summary plot
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for j in range(n_circles):
    ax.plot(time_points, R_hist_list[j], lw=2, label=f"#{j+1}")

# 🔹 Final: enforce full time and radius ranges
ax.set_xlim(0.0, T_final)
max_R_seen_final = max(max(R_hist_list[j]) for j in range(n_circles) if R_hist_list[j])
ax.set_ylim(0.0, 1.1*max_R_seen_final if max_R_seen_final > 0 else 1.0)

ax.set_xlabel("t"); ax.set_ylabel("R_j(t)")
ax.set_title("3D droplet radii with lipolysis")
ax.legend(loc='best', fontsize=9, frameon=True)

plt.show()
