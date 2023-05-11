# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:38:00 2020

@author: Jordan Day
"""


from scipy import sparse
from tqdm import tqdm
from Object_classes import *
from Plotting_jd import *
import os
from pathlib import Path
import solver


nx = 300
ny = 300
nz = 1
Lx = 0.32  # length in m
Ly = 0.45  # length in m
Lz = 0.25  # length in m
Lx = 1  # length in m
Ly = 1  # length in m
Lz = 1  # length in m
hrt = 12  # hrt set in any given time units TC is the conversion to seconds
hrt *= TC_hour  # The hydraulic retention time converted into seconds
baffle_length = 91 / 100  # This is the fraction of the tank a baffle takes up in x
baffle_pairs = 6 # Number of baffle pairs (RHS+LHS) = 1 pair.

file_name = 'Y_28_s_500_RT_7_days_HRT_2_baffles_6' # File name for saving model data
# Baffle param

# Runtime for reactor system
RT = 1
RT *= TC_day#
dt_max = 8
k = 20  # Store data every k min, 1 is every min, 2 every 2 min
D = 1 / hrt
stream = True#
anode_side = 'all'  # can either be 'all', 'influent', 'effluent'

dx = Lx / nx
dy = Ly / ny
x = np.linspace(0, Lx, nx).T
y = np.linspace(0, Ly, ny).T
[yy, xx] = np.meshgrid(np.linspace(dy / 2, Ly - dy / 2, ny), np.linspace(dx / 2, Lx - dx / 2, nx))

system = domain(Lx, Ly, Lz, nx=nx, ny=ny)

flux = (Lx * Ly) / hrt  # The "area flux" through the system
nxy = nx * ny
nxy_one = (nx + 1) * (ny + 1)
psi = np.zeros((nx + 1, ny + 1))  # This is the stream function
boundary = np.zeros((nx + 1, ny + 1))  # We set this to 1 on all boundary points
boundary[0, :] = 1
boundary[-1, :] = 1
boundary[:, 0] = 1
boundary[:, -1] = 1

edges = boundary
psi[0, 0:ny + 3] = flux
psi[:, -1] = flux
psi, boundary, in_out_points, in_start, out_start = system.influent_effluent_regions(baffle_pairs, baffle_length,
                                                                                     14*dy, psi, boundary, flux)#0.06

bdata = psi[boundary == 1]

file_a = 'hrt' + str(hrt).replace('.', '_') + '_nx' + str(nx) + '_ny' + str(ny) + '_Lx' + str(Lx) + '_Ly' + str(
    Ly) + '_pairs' + str(baffle_pairs) + '_width' + str(np.round(baffle_length, decimals=1)).replace('.',
                                                                                                     '_') + '.csv'
file_x = 'Ux_' + file_a
file_y = 'Uy_' + file_a
data_folder = Path(os.getcwd(), "Output", "Velocity")

try:
    ux = np.genfromtxt(data_folder / file_x, delimiter=',')
    uy = np.genfromtxt(data_folder / file_y, delimiter=',')
except:
    psi, ux, uy, resid = solver.steady_state(boundary, psi, nx, ny, dx, dy,
                                             error=1e-6)  # Using function to determine steady state
    np.savetxt(data_folder / file_x, ux, fmt='%.18e', delimiter=',')
    np.savetxt(data_folder / file_y, uy, fmt='%.18e', delimiter=',')
    ux_i = 0.5 * (ux[0:nx, :] + ux[1:nx + 1, :])
    uy_i = 0.5 * (uy[:, 0:ny] + uy[:, 1:ny + 1])
    speed = np.sqrt(ux_i.T ** 2 + uy_i.T ** 2)
    lw = 10 * speed / speed.max()
    fig = plt.figure(figsize=(12, 12))
    plt.streamplot(xx.T, yy.T, ux_i.T, uy_i.T, color=speed, density=1.5, linewidth=1.5, cmap='coolwarm',
                   arrowstyle='-|>')  # coolwarm')
    plt.colorbar()
    plt.xlim((0, Lx))
    plt.ylim((0, Ly))


# %%
external = np.zeros(boundary.shape)
external[0, :] = 1
external[-1, :] = 1
external[:, 0] = 1
external[:, -1] = 1
internal = boundary - external
bio_loc = np.zeros(boundary.shape)
bio_loc[:, 0:ny] += internal[:, 1:ny + 1]
bio_loc[:, 1:ny + 1] += internal[:, 0:ny]
bio_loc = bio_loc[0:nx, 0:ny]
if baffle_length == 0 and baffle_pairs == 0:
    bio_loc[1:-1, -2] = 1
    bio_loc[1:-1, 1] = 1

positional = np.nonzero(np.mean(bio_loc, 0))
switch = np.zeros(bio_loc.shape)
if anode_side == 'influent':
    switch[:, positional[0][0:20:2]] = 1  # :2 to alternate front and back on off 1:20:2 is back, 0:20:2 is front
    bio_loc *= switch
    print('influent facing anodes are active')
elif anode_side == 'effluent':
    switch[:, positional[0][1:20:2]] = 1  # :2 to alternate front and back on off 1:20:2 is back, 0:20:2 is front
    bio_loc *= switch
    print('effluent facing anodes are active')
else:
    anode_side = 'all'
    print('All anodes are active')

bio_number = np.count_nonzero(bio_loc)
# %%

anode_numbers = np.count_nonzero(np.mean(bio_loc, 0))

# Determine anode area based on biofilm and baffle length
Ai = dx * Lz
A = baffle_length * nx * Ai
# A = Lx * Lz #anode area
# Ai = A/nx # area per cell

Vol = Lx * Ly * Lz * 1e3  # Volume in
Voli = dx * dy * Lz * 1e3  # Local volume in L

convert_m2_l = Ai / Voli
convert_l_m2 = Voli / Ai

za = MicrobialPopulation(3000 * np.ones(bio_number)  # np.random.normal(loc = 1000,scale = 10,size = (nx,2)) #initial
                         , 7.9 / TC_day  #7.9 / TC_day  # consumption
                         , 3*1.2 / TC_day  # growth
                         , 0.02  # decay
                         , 20  # sub monod const
                         , 'Anodophilic'  # Defining anodophilic species as class
                         , 5000
                         , mediator_monod=0.2 * 1)


s = Substrate(100 * np.ones((nx, ny)), influent=1000, diffusion=1e-9, name='Acetate')
s.current = s.update_influent(baffle_pairs, in_out_points, ny)

m_total = 1  # mg-M / mg-Z Initial mediator

mox = Parameters(0.99 * m_total * np.ones(bio_number), name='Oxidised Mediator')

mred = Parameters(m_total - mox.initial, name='Reduced Mediator')
Ym = 22#.75  # mg-M /mg -s 36#
m = 2  # mol-e / mol-M
gamma = 663400  # mg-M/mol-M
T = 298  # K
j0 = 1e-2  # 1e-2-> almost identical to 1, but much faster run times
BV_full = False

j_min = 1.60
j_max = 1.60  # 1.34#0.64
##############################################
# changed here and j_0 ###
E_min_scaled = j_min * (A * 500)  # /(R*T/(m*F*j0))Anode area times sum of res
E_max_scaled = j_max * (A * 500)  # /(R*T/(m*F*j0)) Anode area times sum of res

#############################################
j_test = 1.3736263736263736#1.5  # .4#2.4#1.6#2.4/10  # 40 j0 = 1e-4
# Full BV stable for hrt = 2, 6, j0 = 1e-4 , J_test = 1.4

Rin = Parameters(0, minimum=7, maximum=7, k=0.006 * A / Vol, name='Internal Resistance')
Rex = 1  # ohm
E_test = 10  # j_test * (A * (Rin.minimum+Rex)) # j_test*(R*T/(m*F*j0)+500*A*(0.92/0.08)/(0.92/0.08))
E = Parameters(0, minimum=E_test, maximum=E_test, k=0.0006, name='Voltage')


pref = gamma / (m * F)
I = Parameters(0, name='Current (Not to be confused with current value)')
s_blank = np.zeros((bio_number, 5000))
ij = 0
t = GeneralVariable(0, name='Time')
Rin.current = Rin.minimum  # +(Rin.maximum - Rin.minimum)*np.exp(-Rin.k*sum(Za.initial)/nx) + Rex
Rin.storage[0] = Rin.current

# setting up locations for biofilm
# bio_loc = np.zeros((nx,ny))
bio_upper = np.zeros((nx, ny))
bio_lower = np.zeros((nx, ny))
bio_lower[:, -2] = 1  # used for plotting
bio_upper[:, 1] = 1
consump = np.zeros((nx, ny))  # setting consumption array
med_dist = np.zeros(consump.shape)

ux_max = np.max(ux)  # max velocity based on steady state
uy_max = np.max(uy)  # max vel from steady state
# Creating sparse matrix for biofiolm diffusion]
positions = [-1, 0, 1]
diag_x = np.array([[1 / (dx ** 2)], [-2 / (dx ** 2)], [1 / (dx ** 2)]]).repeat(nx, axis=1)
diag_y = np.array([[1 / (dy ** 2)], [-2 / (dy ** 2)], [1 / (dy ** 2)]]).repeat(ny, axis=1)
Dx = sp.sparse.spdiags(diag_x, positions, nx, nx)  # d/dx mat Alternate approach to using diffuse_S array
Dy = sp.sparse.spdiags(diag_y, positions, ny, ny)  # d/dy mat
kappa_bio = 1e-12  # diffusion rate for biofilm
Dx_bio = sp.sparse.spdiags(diag_x, positions, nx, nx).tolil()
Dx_bio[0, -1] += 1 / (dx ** 2)  # Periodic Boundary Conditions
Dx_bio[-1, 0] += 1 / (dx ** 2)  # Periodic BC
# Dx_bio[0,-1] += 1/(dx**2) # non peridoic bc
# Dx_bio[0,-1] += 1/(dx**2) # Need to fix
Dx_bio *= kappa_bio  # setting up diffusion array for biofilm
Dx_bio = Dx_bio.tocsr()

Bx = sp.sparse.spdiags(diag_x, positions, nx, nx)  # tolil()
By = sp.sparse.spdiags(diag_y, positions, ny, ny)
Iy = sp.sparse.eye(ny)
Ix = sp.sparse.eye(nx)
Diffuse_s = (sp.sparse.kron(Iy, Bx) + sp.sparse.kron(By, Ix)).tolil()

bio_diffusion_x = sp.sparse.kron(Iy, Bx).tolil()
temp_location = np.zeros((nx + 1, ny + 1))
temp_location[:-1, :-1] = bio_loc

for ii in np.arange(nxy):
    ix = int(ii % nx)
    iy = int(np.floor(ii / nx))
    jj = iy * (nx + 1) + ix
    if boundary[ix, iy] * boundary[ix, iy + 1] == 1:  # Boundary on left
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dx ** 2)  #
        if ix != 0:
            Diffuse_s[ii, ii - 1] = 0
    if boundary[ix + 1, iy] * boundary[ix + 1, iy + 1] == 1:  # Boundary on right
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dx ** 2)
        if ix != nx - 1:
            Diffuse_s[ii, ii + 1] = 0

    if boundary[ix, iy] * boundary[ix + 1, iy] == 1:  # Boundary below
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dy ** 2)
        if iy != 0:
            Diffuse_s[ii, ii - nx] = 0

    if boundary[ix, iy + 1] * boundary[ix + 1, iy + 1] == 1:  # Boundary above
        Diffuse_s[ii, ii] = Diffuse_s[ii, ii] + 1 / (dy ** 2)
        if iy != ny - 1:
            Diffuse_s[ii, ii + nx] = 0
    if temp_location[ix, iy] * temp_location[ix + 1, iy] == 0:
        bio_diffusion_x[ii, ii] += 1 / (dx ** 2)
    if ix == 0:
        bio_diffusion_x[ii, ii] += 1 / (dx ** 2)
    if ix != 0:
        if temp_location[ix - 1, iy] * temp_location[ix, iy] == 0:
            bio_diffusion_x[ii, ii] += 1 / (dx ** 2)

    if temp_location[ix, iy] + temp_location[ix + 1, iy] == 0:
        bio_diffusion_x[ii, ii] -= 1 / (dx ** 2)

Diffuse_s = Diffuse_s.tocsr()
LU = sp.sparse.linalg

bio_diffusion_x = 1e-6 * bio_diffusion_x.tocsr()

za.calculate_positional_distribution(bio_loc)
temp = bio_diffusion_x.dot(np.reshape(za.positional_distribution.T, nxy))
temp = np.reshape(temp.T, (ny, nx)).T
temp[bio_loc != 1] = 0  # Deals with mass produced outside of biofilm! (temp fix)

del temp_location

s.storage[:, :, 0] = s.initial
s.current[:, :] = s.initial  # This line causes s.initial to be linked to s.now

dt = min(dt_max, 1 / (ux_max / dx + uy_max / dy), (dx ** 2 * dy ** 2) / (
        2 * s.diffusion * (dx ** 2 + dy ** 2)))
# dt = np.floor(dt*100)/100

ii = 0
rk = np.array([[0, 1 / 2, 1 / 2, 1], [0, 1, 1, 1], [1, 2, 2, 1]]).T

# These have been taken out of loop as they remain constant and independent of Z

bound_out = np.zeros(s.current.shape)
bound_in = np.zeros(s.current.shape)
bound_out[-1, :] = 1
bound_out[:, -1] = 1
bound_out[:, 0] = 1
bound_in[-2, :] = 1
bound_in[:, -2] = 1
bound_in[:, 1] = 1

E.current = E.minimum
Rin.current = Rin.minimum
Rsig = Rin.current + Rex

mred.current = m_total - mox.current
eta_conc = R * T / (m * F) * np.log(m_total / mred.current)

med_dist[bio_loc == 1] = Ai * mred.current / mox.current
summation = np.array([Rsig * np.sum(med_dist, 0), ] * nx)
summation_shaped = summation[bio_loc == 1]
j = (mred.current / mox.current * (E.current - eta_conc)) / (
        R * T / (m * F * j0) + summation_shaped)
del eta_conc

print("System will simulate a {} baffle system with a fluid HRT of {} hours and bio runtime of {} days".format(
    2 * baffle_pairs, hrt / TC_hour, RT / TC_day))

start_time_bio = time.time()
storage_steps = int(k * 60 / dt)
s.current = s.update_influent(baffle_pairs, in_out_points, ny)
total_mass_in = (ux[0, (in_start - 1):in_start + in_out_points - 1] * s.initial[0, (in_start - 1):in_start + in_out_points - 1] * Voli / dx).sum()*dt
total_mass_out = 0
total_removed = 0
total_remaining = 0
total_time = time.time()
pbar = tqdm(total=101, desc="Progress of simulation", ncols=100, )

while t.current < RT + 10:
    ii += 1
    lt = time.time()
    irk = 0
    consump *= 0  # Reset consumption
    while irk < 4:  # replaced with while loop to allow for adaptive timesteps
        if irk == 0:
            za.intermediate = za.current
            s.intermediate = s.current
            mox.intermediate = mox.current
            mred.intermediate = m_total - mox.intermediate
        else:
            za.update_intermediate(rk, irk, dt)
            s.update_intermediate(rk, irk, dt)
            mox.update_intermediate(rk, irk, dt)
            mred.intermediate = m_total - mox.intermediate

        if (mox.current + (dt / 6) * mox.ddt2 > m_total).any() or (
                mox.current + rk[irk, 0] * dt * mox.ddt1 > m_total).any():
            # If over estimating rk4 loop is reset with smaller timestep
            irk = 0
            dt *= 0.5
            continue

        local_s = np.reshape(s.intermediate[bio_loc == 1], bio_number)
        substrate_mean_surface = anode_surface_sum(Ai * local_s, bio_loc) / A

        j, eta_act = current_density_inter(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=BV_full)
        I_anode = anode_surface_sum_repeated(j * Ai, bio_loc)
        mediator_current_density_term = I_anode / A

        # local_s = np.reshape(s.intermediate[bio_loc == 1], (bio_number))
        za.update_growth_and_consumption(local_s, mox.intermediate)
        za.first_timestep()
        za.second_timestep(rk, irk)
        mox.ddt1 = -Ym * za.consumption + pref * j / za.intermediate
        mox.second_timestep(rk, irk)
        s.calculate_advection(ux, uy, dx, dy)  # Advection is slowest process
        s.calculate_diffusion(Diffuse_s)  # diff is second slowest
        s.calculate_consumption(za, biofilm_location=bio_loc, convert_m2_l=convert_m2_l)  # rapid
        s.first_timestep()  # 
        s.second_timestep(rk, irk)

        irk += 1  # move forward in rk4 loop
        if irk == 4 and (mox.current + (dt / 6) * mox.ddt2 > m_total).any():
            irk = 0
            dt *= 0.5
            continue

    za.update_current(dt)
    s.update_current(dt)
    mox.update_current(dt)
    mred.current = m_total - mox.current
    j, eta_act, eta_conc = current_density(E, Rin, Rex, m_total, mred, mox, bio_loc, Ai, R, T, m, F, j0, full=BV_full)
    I.current = np.sum(Ai * j)
    t.current += dt
    dt = min(dt_max, dt * 2, 1 / (ux_max / dx + uy_max / dy), (dx ** 2 * dy ** 2) / (
            2 * s.diffusion * (dx ** 2 + dy ** 2)))  # increase timestep up to 2 or double previous timestep
    s.current = s.update_influent(baffle_pairs, in_out_points, ny)
    za.check_mass()



    if ii % storage_steps == 0 or ii == 1:  # round(t.now,2)%(k*60) == 0 : #Storage of data
        ij += 1
        za.storage[:, ij] = za.current
        # Xm.storage[:, ij] = Xm.current
        mox.storage[:, ij] = mox.current
        mred.storage[:, ij] = mred.current
        I.storage[ij] = I.current
        t.storage[ij] = t.current
        s.storage[:, :, ij] = s.current
        increase = round((t.current - t.storage[ij - 1]) / (RT + 20) * 100, 1)
        pbar.update(round(increase, 1))
#
save_data_classes(file_name,['Output','temp'],s,za,mox,mred,j,t)
plot_positional_data(x, j, bio_loc, new_fig=True)

plt.figure(figsize=(14, 10))
plt.subplot(221)
plot_contour(xx, yy, s.current)
plt.subplot(222)
plot_positional_data(x, j, bio_loc, side='right', title='Positional current density A/m^2')
plt.subplot(223)
plot_time_series(t.storage[0:ij + 1] / TC_day, I.storage[0:ij + 1] / (20 * A), linelabel='Current density over time')
plt.subplot(224)
plot_positional_data(x, za.current, bio_loc, side='right', title='Positional biomass mg/m^2')