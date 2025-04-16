#import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt

def get_Fg(m1, m2, x1, y1, x2, y2): # get force of central mass on probe
    dx = x2 - x1
    dy = y2 - y1
    r = sqrt(dx**2 + dy**2)
    
    force = G * m1 * m2 / r**2
    fx = -force * dx / r
    fy = -force * dy / r
    return fx, fy

def acceleration(m, x, y): # get accel components for rk4
    fx, fy = get_Fg(m_center, m, 0, 0, x, y)
    return fx / m, fy / m

def rk4(m,vx,vy,x,y): # 4th order runge-kutta method
    ax1, ay1 = acceleration(m, x, y)

    k1vx, k1vy = ax1 * dt, ay1 * dt
    k1x, k1y = vx * dt, vy * dt

    ax2, ay2 = acceleration(m, x + k1x / 2, y + k1y / 2)
    k2vx, k2vy = ax2 * dt, ay2 * dt
    k2x, k2y = (vx + k1vx / 2) * dt, (vy + k1vy / 2) * dt

    ax3, ay3 = acceleration(m, x + k2x / 2, y + k2y / 2)
    k3vx, k3vy = ax3 * dt, ay3 * dt
    k3x, k3y = (vx + k2vx / 2) * dt, (vy + k2vy / 2) * dt

    ax4, ay4 = acceleration(m, x + k3x, y + k3y)
    k4vx, k4vy = ax4 * dt, ay4 * dt
    k4x, k4y = (vx + k3vx) * dt, (vy + k3vy) * dt

    new_vx = vx + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
    new_vy = vy + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6
    new_x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    new_y = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6

    return new_vx, new_vy, new_x, new_y

# constants
G = 6.67430e-11 # gravitational constant
stage = 1 # 1 is before transfer, 2 is during transfer, 3 is after transfer

m_center = 5.972e24 # central mass (kg)
m_probe = 358 # probe mass(kg)
dt = 3600*2 # timestep in s
R = 384400e3 # initial orbital radius in m
R_prime = R*2 # final orbital radius in m

# initial probe conditions
vi = 1023 # initial speed in m/s
px, py = R, 0 # probe x and y positions in m
vx, vy = 0, vi # probe x and y velocities in m/s

# probe position histories
hx, hy = [], []

# calculate boost speeds
mu = G*m_center
del_v = sqrt(mu/R)*(sqrt((2*R_prime)/(R+R_prime))-1)
del_v_prime = sqrt(mu/R_prime)*(1-sqrt((2*R)/(R+R_prime)))

R_max = max((R,R_prime))

# figure setup
fig, ax = plt.subplots(figsize=(7, 8))
fig.canvas.manager.set_window_title('Hohmann Transfer Orbit - Press R to Restart')
ax.set_xlim(-R_max * 1.1, R_max * 1.1)
ax.set_ylim(-R_max * 1.1, R_max * 1.1)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks([]); plt.yticks([])
ax.set_aspect('equal')
ax.set_facecolor('k')

# patch for central mass
center = plt.Circle((0,0),R//10,ec='tab:blue')
ax.add_patch(center)

# patches for circular orbits at R and R'
R_trace = plt.Circle((0,0),R,ec='tab:gray',fill=False,ls=(5,(4,2)))
R_prime_trace = plt.Circle((0,0),R_prime,ec='tab:gray',fill=False,ls=(5,(4,2)))
ax.add_patch(R_trace)
ax.add_patch(R_prime_trace)

# probe path traces during each stage using a different color
trail3, = ax.plot([], [], color='tab:green', marker=',')
trail2, = ax.plot([], [], color='tab:orange', marker=',')
trail1, = ax.plot([], [], color='tab:red', marker=',')

# marker for probe position
probe, = ax.plot([], [], color='w', marker='o', ms=3)

def restart(event): # restart when 'r' key is pressed
    global stage, px, py, vx, vy, hx, hy
    if event.key == 'r':
        stage = 1
        px, py = R, 0
        vx, vy = 0, vi 
        hx, hy = [], []
        trail1.set_data([],[])
        trail2.set_data([],[])
        trail3.set_data([],[])

def update(frame): # update each frame
    global px, py, vx, vy, hx, hy, stage

    vx, vy, px, py = rk4(m_probe, vx, vy, px, py) # get new velocities and positions

    plt.title(f"Stage {stage}\n$R$ = {int(R//1e3)}km, $R'$ =  {int(R_prime//1e3)}km\n\
$∆v$ = {int(del_v)}m/s, $∆v'$ = {int(del_v_prime)}m/s")

    # update histories
    hx.append(px)
    hy.append(py)

    # plot new probe position
    probe.set_data([px], [py])  
    
    # if the probe has made 1/2 of the circular orbit at R, do first boost and enter stage 2
    if stage == 1:
        trail1.set_data(hx, hy)
        if px < 0 and abs(vx) < 10:
            stage = 2
            vy -= del_v
    
    # if the probe has made 1/2 of the eliptical orbit, do second boost and enter stage 3
    elif stage == 2: 
        trail2.set_data(hx, hy)
        if abs(sqrt(px**2+py**2) - R_prime) < 1e5:
            stage = 3
            vy += del_v_prime

    elif stage == 3:
        trail3.set_data(hx, hy)
    
    return trail3, trail2, trail1, probe

# animation function
ani = animation.FuncAnimation(fig, update, frames=5, interval=1, blit=False)
cid = fig.canvas.mpl_connect('key_press_event', restart)
plt.tight_layout()
plt.show()
