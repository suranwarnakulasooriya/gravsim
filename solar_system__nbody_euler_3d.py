# dependencies
from math import pi, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from collections import deque
from colors import *
from os.path import basename

## <=> value can be changed

## simulation params
G = 6.67430e-11 ## m^3/kgs^2
dt = 60*60*24 ## timestep in s
chosen_body_indices = [0,1,2,3,5] ## indices of bodies to be simulated, put [] to select all bodies
approx_method = 'euler' ## choose name of approx method (currently only euler is working properly)

## visual params
max_draw_r = 10 ## pixel radius of most massive body
scale_exaggeration = 11 ## default 3
draw_trails = True ##
kill_if_bodies_leave = False ## terminate the program automatically if a body flies far off screen
window_scale = 1.1 ## size of plt window as a multiple of the largest initial distance

## all body properties
##   simulation relevant
masses = [1.9891e30,0.330e24,4.87e24,5.972e24,7.342e22,0.642e24,1.060e16,1.51e15,
        1898e24,568e24,86.8e24,102e24] ## kg
positions = [[0,0,0],[57.9e9,0,0],[108.2e9,0,0],[149.6e9,0,0],[149.6e9+0.384e9,0,0],[228e9,0,0],[228e9+9400e3,0,0],[228e9+23460e3,0,0],
        [778.5e9,0,0],[1432e9,0,0],[2867e9,0,0],[4515e9,0,0]] ## m
vels = [[1e-100,0,0],[0,47.4e3,0],[0,35e3,0],[0,29.8e3,0],[0,29.8e3+1023,0],[0,24.1e3,0],[0,24.1e3+2138,0],[0,24.1e3+1.3513e3,0],
        [0,13.1e3,0],[0,9.7e3,0],[0,6.8e3,0],[0,5.4e3,0]] ## m/s
##   visual
densities = [1410,5429,5243,5514,3340,3934,1861,1465,1326,687,1270,1638] ## kg/m^3
colors = [orn,wht,ylw,blu,wht,red,grn,mgt,orn,mgt,cyn,blu] ##
names = ['sun','mercury','venus','earth','moon','mars','phobos','deimos','jupiter','saturn','uranus','neptune'] ##

# =====================================================================================================================

# remove unsimulated bodies from property lists
rejected_body_indices = [i for i in range(len(masses)) if i not in chosen_body_indices] if chosen_body_indices else []
for e,i in enumerate(rejected_body_indices):
    masses.pop(i-e); positions.pop(i-e); vels.pop(i-e)
    densities.pop(i-e); colors.pop(i-e); names.pop(i-e)

# derived from body properties
n = len(masses)
distances = [sqrt(x**2+y**2+z**2) for x,y,z in positions] # distances of all bodies from origin
d_max = max(distances[:n])
speeds = [sqrt(vx**2+vy**2+vz**2) for vx,vy,vz in vels]
if draw_trails: # trail lenghts and positions histories only matter if trails are being drawn
    # naive trail length estimation using orbit circumference
    trail_lengths = [1.95*pi*distances[i]/(speeds[i]*dt) for i in range(n)]
    histories = [deque(maxlen=int(trail_lengths[i])) for i in range(n)] # list of past positions capped @ trail lengths
# "real" radii determined by mass and density but inflated by scale_exaggeration
radii = [(masses[i]/((4/3)*pi*densities[i]))**(1/scale_exaggeration) for i in range(n)]
r_max = max(radii)
# scale displayed radii based on given limits
draw_radii = [max(1,max_draw_r*(radii[i]/r_max)) for i in range(n)]
del distances, speeds, radii, r_max

def get_Fg(m1, m2, x1, y1, z1, x2, y2, z2): # return fx, fy, fz on m2 by m1
    dx = x2 - x1; dy = y2 - y1; dz = z2 - z1
    r = sqrt(dx**2 + dy**2 + dz**2)
    force = -G * m1 * m2 / r**2
    fx = force * dx / r; fy = force * dy / r; fz = force * dz / r
    return fx, fy, fz

def net_accel_euler(i) -> None: # update list of net accels for given body index and all indices after it (bilateral)
    for j in range(i+1,n):
        m1 = masses[j]; m2 = masses[i] # get masses
        # get force components, accelerations are computed by dividing by respestive masses
        fx, fy, fz = get_Fg(m1,m2,positions[j][0],positions[j][1],positions[j][2],
                            positions[i][0],positions[i][1],positions[i][2])
        accels[i] = [accels[i][0] + fx/m2, accels[i][1] + fy/m2, accels[i][2] + fz/m2] # add accel components to index i
        accels[j] = [accels[j][0] - fx/m1, accels[j][1] - fy/m1, accels[j][2] - fz/m1] # subtract from other index (equal and opposite)
    return

def euler(i,m,vx,vy,vz,x,y,z): # euler's method
    ax, ay, az = accels[i] # get accel components
    new_vx = vx + ax * dt; new_vy = vy + ay * dt; new_vz = vz + az * dt # update velocities
    new_x = x + new_vx * dt; new_y = y + new_vy * dt; new_z = z + new_vz * dt # update positions
    return new_vx, new_vy, new_vz, new_x, new_y, new_z

def rk4(i, m, vx, vy, x, y): # 4th order runge-kutta method
    def acceleration(x, y): # get net acceleration on body i from all other bodies
        ax, ay = 0, 0
        for j in range(n):
            if i != j: # exclude self 
                fx, fy = get_Fg(masses[j], m, positions[j][0], positions[j][1], x, y)
                ax += fx / m; ay += fy / m
        return ax, ay

    # k1
    ax1, ay1 = acceleration(x, y)
    k1vx, k1vy = ax1 * dt, ay1 * dt
    k1x, k1y = vx * dt, vy * dt

    # k2
    ax2, ay2 = acceleration(x + k1x / 2, y + k1y / 2)
    k2vx, k2vy = ax2 * dt, ay2 * dt
    k2x, k2y = (vx + k1vx / 2) * dt, (vy + k1vy / 2) * dt

    # k3
    ax3, ay3 = acceleration(x + k2x / 2, y + k2y / 2)
    k3vx, k3vy = ax3 * dt, ay3 * dt
    k3x, k3y = (vx + k2vx / 2) * dt, (vy + k2vy / 2) * dt

    # k4
    ax4, ay4 = acceleration(x + k3x, y + k3y)
    k4vx, k4vy = ax4 * dt, ay4 * dt
    k4x, k4y = (vx + k3vx) * dt, (vy + k3vy) * dt

    # weighted sum
    new_vx = vx + (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6
    new_vy = vy + (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6
    new_x = x + (k1x + 2*k2x + 2*k3x + k4x) / 6
    new_y = y + (k1y + 2*k2y + 2*k3y + k4y) / 6

    return new_vx, new_vy, new_x, new_y

approx_methods = {'euler': euler,'rk4':rk4}
approx_method = approx_methods.get(approx_method, euler)  # default to euler if invalid

# plt figure setup
fig = plt.figure(figsize=(8,8),num=basename(__file__)[:-3])
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim(-d_max * window_scale, d_max * window_scale)
ax.set_ylim(-d_max * window_scale, d_max * window_scale)
ax.set_zlim(-d_max * window_scale, d_max * window_scale)

ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
ax.set_axis_off()
ax.set_title(f"approx. method: {approx_method.__name__} | ∆t: {dt}s")
ax.set_aspect('equal')
ax.set_facecolor(blk)

# create plots for each body and corresponding trail if specified
bodies = []; trails = []
for i in range(n):
    if draw_trails: trails.append(ax.plot([], [], [],color=crm, marker=',',alpha=0.1)[0])
    bodies.append(ax.plot([], [], [], colors[i], marker='o', ms=draw_radii[i], label=names[i])[0])

def update(frame): # ran for every tick
    global accels
    accels = [[0,0,0]]*n # reset accelerations of all bodies
    
    if approx_method == euler:
        for i in range(n-1): #    get net accel components of all bodies using bilateral method,
            net_accel_euler(i)  # rk4 computes net accel components automatically
    
    for i in range(n):
        # update velocities and positions according to given approx_method
        vels[i][0], vels[i][1], vels[i][2], positions[i][0], positions[i][1], positions[i][2] = approx_method(
            i, masses[i], vels[i][0], vels[i][1], vels[i][2], positions[i][0], positions[i][1], positions[i][2])

        bodies[i].set_data([positions[i][0]], [positions[i][1]]) # redraw body plots
        bodies[i].set_3d_properties([positions[i][2]]) # z positions

        if draw_trails: # update histories and trail plots
            histories[i].append([positions[i][0],positions[i][1],positions[i][2]])
            trails[i].set_data([p[0] for p in histories[i]], [p[1] for p in histories[i]])   
            trails[i].set_3d_properties([p[2] for p in histories[i]]) 

    if kill_if_bodies_leave: # terminate the program if a body flies far off screen (double the window size)
        for x,y in positions:
            if abs(x) > d_max*window_scale*2 or abs(y) > d_max*window_scale*2: exit('bodies left screen.')

    return trails+bodies if draw_trails else bodies

def main(): # mainloop
    ani = animation.FuncAnimation(fig, update, frames=5, interval=1, blit=False)
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.tight_layout()
    try: plt.show()
    except KeyboardInterrupt: print(' KeyboardInterrupt.'); exit()

if __name__ == '__main__': main()